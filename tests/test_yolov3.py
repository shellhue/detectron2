import logging
import unittest
import torch
import math
from projects.Yolov3.yolov3 import Yolov3
from projects.Yolov3.yolov3 import add_yolov3_config
from detectron2.config import get_cfg
from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes

logger = logging.getLogger(__name__)

class TestYolov3(unittest.TestCase):
    def test_get_ground_truth(self):
        cfg = get_cfg()
        add_yolov3_config(cfg)

        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.BACKBONE.NAME = "build_darknet_fpn_backbone"

        cfg.MODEL.DAKRNET.NORM = "BN"
        cfg.MODEL.DAKRNET.STEM_OUT_CHANNELS = 32
        cfg.MODEL.DAKRNET.OUT_FEATURES = ["s3", "s4", "s5"]
        cfg.MODEL.DAKRNET.DEPTH = 53
        cfg.MODEL.DAKRNET.NUM_CLASSES = 0

        cfg.MODEL.YOLOV3.NUM_CLASSES = 80
        cfg.MODEL.YOLOV3.NORM = "BN"
        cfg.MODEL.YOLOV3.IN_FEATURES = ["p3", "p4", "p5"]
        cfg.MODEL.YOLOV3.HEAD.OUT_CHANNELS = [256, 512, 1024]

        cfg.MODEL.ANCHOR_GENERATOR.NAME = "YoloAnchorGenerator"
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]]

        cfg.MODEL.GRID_GENERATOR.NAME = "DefaultGridGenerator"

        cfg.MODEL.STRIDE_GENERATOR.NAME = "DefaultStrideGenerator"

        cfg.MODEL.DarknetFPN.IN_FEATURES = ["s3", "s4", "s5"]

        cfg.MODEL.DarknetFPN.OUT_CHANNELS = [128, 256, 512]

        model = Yolov3(cfg)

        img_height = 800
        img_width = 960
        predict_sizes = [torch.Size([img_height // 8, img_width // 8]), torch.Size([img_height // 16, img_width // 16]), torch.Size([img_height // 32, img_width // 32])]
        anchor_sizes = [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]]
        strides = [8, 16, 32]
        device = "cpu"
        instances = Instances((img_height, img_width))

        gt = torch.tensor([[0, 0, 11, 14], [305, 279, 365, 324], [58, 108.8, 210.4, 307.5]], dtype=torch.float32)
        instances.gt_boxes = Boxes(gt)
        instances.gt_classes = torch.tensor([1, 20, 56], dtype=torch.long)

        position_target = torch.tensor([[0,  0,  0,  0], [1, (365+305)/2//16, (324+279)/2//16,  1], [2, (210.4+58)/2//32, (307.5+108.8)/2//32,  1]], dtype=torch.int32)
        delta_target = torch.tensor([
            [11/2%8/8,  14/2%8/8,  math.log(11/10+1e-8),  math.log(14/13+1e-8)],
            [(365+305)/2%16/16, (324+279)/2%16/16, math.log((365-305)/62+1e-8), math.log((324-279)/45+1e-8)],
            [(210.4+58)/2%32/32, (307.5+108.8)/2%32/32, math.log((210.4-58)/156+1e-8), math.log((307.5-108.8)/198+1e-8)],
            ])
        
        class_target = torch.tensor([1, 20, 56], dtype=torch.int32)

        stage_stride = [0]
        
        for i, (ps, anchor_size) in enumerate(zip(predict_sizes, anchor_sizes)):
            stage_stride.append(ps[0]*ps[1]*len(anchor_size))
            stage_stride[i+1] += stage_stride[i]
        
        gt_classes, gt_deltas = model.get_ground_truth(predict_sizes,anchor_sizes,  strides, [instances], device)
        gt_mask = gt_classes < (cfg.MODEL.YOLOV3.NUM_CLASSES)
        foreground = gt_mask.nonzero()
        gt_classes_sel = gt_classes[gt_mask]
        gt_deltas_sel = gt_deltas[foreground[...,0], foreground[...,1]]
        
        match_info = torch.zeros((foreground.size()[0], 4), dtype=torch.int)
        for i, (b, idx) in enumerate(foreground):
            s = -1
            idx_o = idx.clone()
            for j, (l, r) in enumerate(zip(stage_stride[:-1], stage_stride[1:])):
                if idx >= l and idx < r:
                    s = j
                    break
            assert s >= 0
            fm_width = predict_sizes[s][1]
            cell_num = len(anchor_sizes[s])
            
            idx -= stage_stride[s]
            c = idx % cell_num
            idx //= cell_num
            h = idx // fm_width
            w = idx % fm_width

            match_info[i, 0] = s
            match_info[i, 1] = w
            match_info[i, 2] = h
            match_info[i, 3] = c
            assert stage_stride[s]+h*cell_num*fm_width+w*cell_num+c == idx_o
        assert torch.equal(match_info.int(), position_target.int())
        assert torch.equal(gt_classes_sel.int(), class_target.int())
        assert torch.allclose(delta_target, gt_deltas_sel)



if __name__ == "__main__":
    unittest.main()