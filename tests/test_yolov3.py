import logging
import unittest
import torch

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
        gt = torch.tensor([[0, 0, 10, 13], [305, 279, 365, 324]], dtype=torch.float32)
        instances.gt_boxes = Boxes(gt)
        instances.gt_classes = torch.tensor([1, 20], dtype=torch.long)
        stage_stride = [0]

        for i, (ps, anchor_size) in enumerate(zip(predict_sizes, anchor_sizes)):
            stage_stride.append(ps[0]*ps[1]*len(anchor_size))
            stage_stride[i+1] += stage_stride[i]
        
        gt_classes, gt_deltas = model.get_ground_truth(predict_sizes,anchor_sizes,  strides, [instances], device)

        num_foreground_nonzero = (gt_classes < (cfg.MODEL.YOLOV3.NUM_CLASSES)).nonzero()
        
        gt_deltas_nonzero = (gt_deltas.sum(dim=-1) != 0).nonzero()
        gt_deltas_sel = gt_deltas[gt_deltas_nonzero[...,0],gt_deltas_nonzero[...,1]]

        print(gt_deltas_sel)
        match_info = torch.zeros((num_foreground_nonzero.size()[0], 4), dtype=torch.int)
        print(num_foreground_nonzero)
        for i, (b, idx) in enumerate(num_foreground_nonzero):
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
        print(match_info)
        
        # idx = gt_deltas_nonzero[..., 1]
        # stage_mask = torch.zeros((len(stage_stride), idx.size()[0]))
        # w = torch.zeros((len(stage_stride), idx.size()[0]))
        # h = torch.zeros((len(stage_stride), idx.size()[0]))
        # a = torch.zeros((len(stage_stride), idx.size()[0]))
        # stage_mask[0][idx < stage_stride[0]] = 1
        # for i in range(len(stage_mask)):
        #     stride = predict_sizes[i][1] * len(anchor_sizes[i])
            
        #     print(stride)
        #     w[i] = idx % stride
        #     h[i] = idx // stride
        #     a[i] = idx % len(anchor_sizes[i])
        # stage_mask[1][(idx >= stage_stride[0]) * (idx < stage_stride[1])] = 1
        # stage_mask[2][(idx >= stage_stride[1]) * (idx < stage_stride[2])] = 1

        # f = torch.cat([h.unsqueeze(-1), w.unsqueeze(-1), a.unsqueeze(-1)], dim=-1) 
        # print(f)
        # print(stage_mask.size())
        # print(f.size())
        # print(gt_classes.size(), gt_deltas.size())

        # for 
        

        assert model is not None



if __name__ == "__main__":
    unittest.main()