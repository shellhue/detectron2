import math
import torch
import numpy as np
from torch import nn
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_anchor_generator
# from detectron2.modeling import build_grid_generator, build_stride_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher

from detectron2.layers import batched_nms


from .yolov3_box_predictor import Yolov3Head
from .grid_generator import build_grid_generator
from .stride_generator import build_stride_generator

@META_ARCH_REGISTRY.register()
class Yolov3(nn.Module):
    def __init__(self, cfg=None):
        super(Yolov3, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_classes = cfg.MODEL.YOLOV3.NUM_CLASSES
        self.norm = cfg.MODEL.YOLOV3.NORM
        self.in_features = cfg.MODEL.YOLOV3.IN_FEATURES
        self.anchors = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        
        # Inference parameters:
        self.score_threshold          = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
        self.nms_threshold            = cfg.MODEL.RETINANET.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

        self.backbone = build_backbone(cfg)

        self.head = Yolov3Head(in_features=self.in_features, 
                               in_channels=[self.backbone._out_feature_channels[f] for f in self.in_features],
                               out_channels=cfg.MODEL.YOLOV3.HEAD.OUT_CHANNELS,
                               num_classes=self.num_classes,
                               num_anchors_per_cell=3, 
                               norm=self.norm)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in self.in_features]
        self.feature_shapes = [backbone_shape[f] for f in self.in_features]
        
        self.anchor_generator = build_anchor_generator(cfg, self.feature_shapes)
        self.grid_generator = build_grid_generator(cfg, self.feature_shapes)
        self.stride_generator = build_stride_generator(cfg, self.feature_shapes)

        # self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

        self.normalizer = lambda x: x / 255.0
        self.to(self.device)
        self.get_conv_bn_modules()
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        fpn_features_dict = self.backbone(images.tensor)
        
        features = [fpn_features_dict[f] for f in self.in_features]
        
        
        device = features[0].device

        predicts = self.head(features)
        predicts_permuted = []
        for p, anchor in zip(predicts, self.anchors):
            n, _, hi, wi = p.size()
            p = p.reshape(n, len(anchor), (self.num_classes + 5), hi, wi)
            p = p.permute(0, 3, 4, 1, 2)
            predicts_permuted.append(p)
        predicts_confidence_logits = [p[..., 4] for p in predicts_permuted]
        predicts_classes_logits = [p[..., 5:] for p in predicts_permuted]
        predicts_anchor_deltas = [p[..., :4] for p in predicts_permuted]
        
        if self.training:
            predict_sizes = [i.size()[-2:] for i in predicts]
            # print("input image size:", images.tensor.size())
            # print("instances:", gt_instances)
            # print("batched_inputs:", batched_inputs)
            # print("predict_sizes: ", predict_sizes)
            # print("predicts_confidence_logits: ", [i.size() for i in predicts_confidence_logits])
            # print("predicts_classes_logits: ", [i.size() for i in predicts_classes_logits])
            # print("predicts_anchor_deltas: ", [i.size() for i in predicts_anchor_deltas])
            # print(self.anchors)
            # print(self.feature_strides)
            # assert False

            gt_classes, gt_anchor_deltas = self.get_ground_truth(predict_sizes, self.anchors, self.feature_strides, gt_instances, device)
            # print(gt_classes.size(), gt_anchor_deltas.size())
            # assert False

            losses = self.losses(gt_classes, gt_anchor_deltas, predicts_confidence_logits, predicts_classes_logits, predicts_anchor_deltas)
            

            return losses
        else:
            strides = self.stride_generator(predicts)
            grids = self.grid_generator(predicts)
            anchors = self.anchor_generator(predicts)

            predicts_confidence_logits = [p.reshape(n, -1) for p in predicts_confidence_logits]
            
            predicts_classes_logits = [p.reshape(n, -1, self.num_classes) for p in predicts_classes_logits]

            predicts_anchor_deltas = [p.reshape(n, -1, 4) for p in predicts_anchor_deltas]

            results = self.inference(predicts_confidence_logits, predicts_classes_logits, predicts_anchor_deltas, anchors, strides, grids, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = self.rescale_box(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results


    def losses(self, gt_classes, gt_anchor_deltas, predicts_confidence_logits, predicts_classes_logits, predicts_anchor_deltas):
        """
        Args:
            gt_classes (Tensor]): An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
            gt_anchor_deltas (Tensor]): An integer tensor of shape (N, R, 4) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        n = predicts_confidence_logits[0].size()[0]

        # reshape to NR
        predicts_confidence_logits = [p.reshape(-1) for p in predicts_confidence_logits]
        predicts_confidence_logits = torch.cat(predicts_confidence_logits, dim=0)
        
        # reshape to NRxK
        predicts_classes_logits = [p.reshape(-1, self.num_classes) for p in predicts_classes_logits]
        predicts_classes_logits = torch.cat(predicts_classes_logits, dim=0)

        # reshape to NRx4
        predicts_anchor_deltas = [p.reshape(-1, 4) for p in predicts_anchor_deltas]
        predicts_anchor_deltas = torch.cat(predicts_anchor_deltas, dim=0)

        # reshape to NR
        gt_classes = [g.reshape(-1, ) for g in gt_classes]
        gt_classes = torch.cat(gt_classes, dim=0)
        
        # reshape to NR
        gt_anchor_deltas = [g.reshape(-1, 4) for g in gt_anchor_deltas]
        gt_anchor_deltas = torch.cat(gt_anchor_deltas, dim=0)

        # truth mask NR
        truth_mask = (gt_classes >= 0) & (gt_classes != self.num_classes)
        
        
        negative_mask = gt_classes == self.num_classes
        num_foreground = truth_mask.sum()
        
        # loss xy
        # print(num_foreground)
        # print(negative_mask.sum())
        # print(gt_classes[truth_mask])
        # print(gt_anchor_deltas[truth_mask])
        loss_xy = torch.pow(predicts_anchor_deltas[truth_mask][..., :2] - gt_anchor_deltas[truth_mask][..., :2], 2)
        loss_xy = loss_xy.sum() / max(1, num_foreground)

        # loss wh
        loss_wh = torch.pow(predicts_anchor_deltas[truth_mask][..., 2:] - gt_anchor_deltas[truth_mask][..., 2:], 2)
        loss_wh = loss_wh.sum() / max(1, num_foreground)

        # loss confidence
        loss_conf = F.binary_cross_entropy_with_logits(predicts_confidence_logits[truth_mask], truth_mask[truth_mask].float()) + F.binary_cross_entropy_with_logits(predicts_confidence_logits[negative_mask], truth_mask[negative_mask].float())
        # loss_conf /= max(1, num_foreground)

        # print(predicts_confidence_logits[negative_mask].size(), truth_mask[negative_mask].size())
        # print("truth_mask num:", truth_mask.sum())
        # print("negative_mask:", negative_mask.sum())
        # print(truth_mask[negative_mask].float())
        # assert False

        # loss class
        loss_classes = F.cross_entropy(predicts_classes_logits[truth_mask], gt_classes[truth_mask].long())
        # loss_classes /= max(1, num_foreground)

        return {
            "loss_xy": loss_xy,
            "loss_wh": loss_wh,
            "loss_conf": loss_conf,
            "loss_classes": loss_classes
        }

    @torch.no_grad()
    def get_ground_truth(self, predict_sizes, anchors, feature_strides, targets, device):
        """
        Args:
            anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
                list of #feature level Boxes. The Boxes contains anchors of
                this image on the specific feature level.
            targets (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.

        Returns:
            gt_classes (Tensor):
                An integer tensor of shape (N, R) storing ground-truth
                labels for each anchor.
                R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
                Anchors with an IoU with some target higher than the foreground threshold
                are assigned their corresponding label in the [0, K-1] range.
                Anchors whose IoU are below the background threshold are assigned
                the label "K". Anchors whose IoU are between the foreground and background
                thresholds are assigned a label "-1", i.e. ignore.
            gt_anchors_deltas (Tensor):
                Shape (N, R, 4).
                The last dimension represents ground-truth box2box transform
                targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
                The values in the tensor are meaningful only when the corresponding
                anchor is labeled as foreground.
        """
        anchor_masks = []
        anchor_sizes = []
        strides = []
        n = len(targets)
        b = 4
        for anchor_sizes_single_level, stride in zip(anchors, feature_strides):
            anchor_masks.append([(i+len(anchor_sizes))
                                 for i in range(len(anchor_sizes_single_level))])
            anchor_sizes.extend(anchor_sizes_single_level)
            strides.extend([stride] * len(anchor_sizes_single_level))

        strides = torch.tensor(strides, device=device, dtype=torch.float)
        anchor_masks = torch.tensor(anchor_masks, device=device, dtype=torch.long)
        anchor_sizes = torch.tensor(anchor_sizes, device=device, dtype=torch.float)
        anchors_boxes = torch.cat([-anchor_sizes / 2.0, anchor_sizes / 2.0], dim=1)
        
        # anchor box shape changed
        anchors_boxes = anchors_boxes.view(1, -1, 4)
        anchors_wh = anchors_boxes[..., 2:] - anchors_boxes[..., :2]
        anchors_area = anchors_wh[..., 0] * anchors_wh[..., 1]
        
        # N_HWA_4
        gt_deltas = []
        # N_HWA
        gt_classes = []
        predict_heights = []
        predict_widths = []
        predict_anchors = []
        scale_offset = [0]
        for size_i, anchor_i in zip(predict_sizes, anchors):
            # feature_level NxHixWixAx4
            hi, wi = size_i
            ai = len(anchor_i)
            gt_classes.append(torch.ones((n, hi, wi, ai), dtype=torch.float, device=device).reshape(n, -1) * self.num_classes)
            gt_deltas.append(torch.zeros((n, hi, wi, ai, b), dtype=torch.float, device=device).reshape(n, -1, b))
            predict_heights.append(hi)
            predict_widths.append(wi)
            predict_anchors.append(ai)
            scale_offset.append(scale_offset[-1] + hi * wi * ai)
        gt_classes = torch.cat(gt_classes, dim=1)
        gt_deltas = torch.cat(gt_deltas, dim=1)

        scale_offset = scale_offset[:-1]
        predict_heights = torch.tensor(predict_heights, dtype=torch.int, device=device)
        predict_widths = torch.tensor(predict_widths, dtype=torch.int, device=device)
        predict_anchors = torch.tensor(predict_anchors, dtype=torch.int, device=device)
        scale_offset = torch.tensor(scale_offset, dtype=torch.int, device=device)

        for idx, instance in enumerate(targets):
            if len(instance) == 0:
                continue
            # stage 1: match gt to anchor
            gt_boxes = instance.gt_boxes.tensor.float()
            gt_boxes_wh = gt_boxes[..., 2:] - gt_boxes[..., :2]
            gt_boxes_center = (gt_boxes[..., 2:] + gt_boxes[..., :2]) / 2.0
            gt_boxes = torch.cat((gt_boxes_wh * -1 / 2.0, gt_boxes_wh / 2.0), dim=-1)
            gt_boxes = gt_boxes.view(-1, 1, 4)
            gt_boxes_wh = gt_boxes_wh.view(-1, 1, 2)
            gt_boxes_area = gt_boxes_wh[..., 0] * gt_boxes_wh[..., 1]
            
            interset_tl = torch.max(gt_boxes[..., :2], anchors_boxes[..., :2])
            interset_br = torch.min(gt_boxes[..., 2:], anchors_boxes[..., 2:])
            interset_wh = interset_br - interset_tl
            interset_area = interset_wh[..., 0] * interset_wh[..., 1]
            
            iou = interset_area / (gt_boxes_area + anchors_area - interset_area)
            matched_idx = torch.argmax(iou, 1)
            
            # stage 2: match gt to stage and anchor
            matched_anchors_wh = anchor_sizes[matched_idx]
            gt_boxes = instance.gt_boxes.tensor
            matched_strides = strides[matched_idx]
            matched_idx = matched_idx.view(-1, 1, 1)
            matched_stage = (matched_idx == anchor_masks.unsqueeze(0)).int()
            stage_anchor_idx = matched_stage.nonzero()[...,1:]
            matched_stage_idx = stage_anchor_idx[..., 0]
            
            # stage 3: assign class and delta
            grid_index = gt_boxes_center // matched_strides.unsqueeze(-1)
            gt_deltas_xy = gt_boxes_center % matched_strides.unsqueeze(-1).float()
            gt_deltas_xy /= matched_strides.unsqueeze(-1)
            gt_deltas_wh = gt_boxes[..., 2:] - gt_boxes[...,:2]
            gt_deltas_wh /= matched_anchors_wh
            gt_deltas_wh = torch.log(gt_deltas_wh)
                
            grid_index = grid_index.int()
            grid_index_x = grid_index.int()[..., 0]
            grid_index_y = grid_index.int()[..., 1]
            matched_final_idx = scale_offset[matched_stage_idx] + \
                predict_widths[matched_stage_idx] * grid_index_y * \
                predict_anchors[matched_stage_idx] + grid_index_x * \
                predict_anchors[matched_stage_idx] + stage_anchor_idx[..., 1]
            gt_classes[idx][matched_final_idx] = instance.gt_classes.float()
            gt_deltas[idx][matched_final_idx] = torch.cat([gt_deltas_xy, gt_deltas_wh], dim=1)
        
        return gt_classes, gt_deltas
    
    def inference(self, box_conf, box_cls, box_delta, anchors, strides, grids, images):
        """
        Arguments:
            box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
            anchors (list[list[Boxes]]): a list of #images elements. Each is a
                list of #feature level Boxes. The Boxes contain anchors of this
                image on the specific feature level.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(anchors) == len(images)
        results = []

        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)
        for img_idx, anchors_per_image in enumerate(anchors):
            image_size = images.image_sizes[img_idx]
            strides_per_image = strides[img_idx]
            grids_per_image = grids[img_idx]
            box_conf_per_image = [box_conf_per_level[img_idx] for box_conf_per_level in box_conf]
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_delta]
            results_per_image = self.inference_single_image(
                box_conf_per_image, box_cls_per_image, box_reg_per_image, anchors_per_image, strides_per_image, grids_per_image, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_conf, box_cls, box_delta, anchors, strides, grids, image_size):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).

        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
            anchors (list[Boxes]): list of #feature levels. Each entry contains
                a Boxes object, which contains all the anchors for that
                image in that feature level.
            image_size (tuple(H, W)): a tuple of the image height and width.

        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_conf_i, box_cls_i, box_reg_i, anchors_i, strides_i, grids_i in zip(box_conf, box_cls, box_delta, anchors, strides, grids):
            # (HxWxA,)
            box_conf_i = box_conf_i.sigmoid_().view(-1, 1)
            # (HxWxA, K)
            box_cls_i = box_cls_i.softmax(-1)
            box_cls_i *= box_conf_i
            # (HxWxAxK)
            box_cls_i = box_cls_i.flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            box_reg_i_center = box_reg_i[..., :2]
            box_reg_i_center = box_reg_i_center.sigmoid_()
            box_reg_i_wh = box_reg_i[..., 2:]
            anchors_i = anchors_i[anchor_idxs]
            anchors_i_wh = anchors_i.tensor[..., 2:] - anchors_i.tensor[..., :2]
            strides_i = strides_i[anchor_idxs].unsqueeze(-1)
            grids_i = grids_i[anchor_idxs]
            # predict boxes
            predicted_boxes_center = (box_reg_i_center + grids_i) * strides_i
            predicted_boxes_wh = torch.exp(box_reg_i_wh) * anchors_i_wh
            predicted_boxes = torch.cat([predicted_boxes_center - predicted_boxes_wh / 2.0, predicted_boxes_center + predicted_boxes_wh / 2.0], dim=-1)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            torch.cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]
        return result


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].float().to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def rescale_box(self, results, output_height, output_width):
        """
        Resize the output instances.
        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.

        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.

        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution the detector sees.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.

        Returns:
            Instances: the resized output from the model, based on the output resolution
        """
        height, width = results.image_size
        scale_x, scale_y = (output_width / width, output_height / height)

        ratio = min(1.0 * height / output_height, 1.0 * width / output_width)
        new_img_h = int(output_height * ratio)
        new_img_w = int(output_width * ratio)
        dw = (width - new_img_w) // 2
        dh = (height - new_img_h) // 2

        results = Instances((output_height, output_width), **results.get_fields())

        if results.has("pred_boxes"):
            output_boxes = results.pred_boxes
        elif results.has("proposal_boxes"):
            output_boxes = results.proposal_boxes
        
        output_boxes.translate(-dw, -dh)
        
        output_boxes.scale(1 / ratio, 1 / ratio)
        output_boxes.clip(results.image_size)
        results = results[output_boxes.nonempty()]

        return results
    
    def get_conv_bn_modules(self):
        """
        for weight convert from original yolo weights file
        """
        modules = []
        darknet_modules, fpn_features_modules = self.backbone.get_conv_bn_modules()
        head_modules = self.head.get_conv_bn_modules()
        modules.extend(darknet_modules)
        for fpn_modules_i, head_modules_i in zip(fpn_features_modules, head_modules):
            modules.extend(fpn_modules_i)
            modules.extend(head_modules_i)
        # for i in modules:
        #     print(i.weight.size())
        # assert False, "debug"
        return modules

