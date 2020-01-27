import torch
from torch import nn
import torch.nn.functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.modeling.backbone.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import BottleneckBlock

from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

def scale_chan(channels, width_mult=1.0, min_chan=16):
    return max(int(channels * width_mult), min_chan)

def conv_bn(inp, oup, stride, width_mult=1.0):
    inp = 3 if inp == 3 else scale_chan(inp, width_mult)
    oup = scale_chan(oup, width_mult)
    conv = Conv2d(inp, oup, 3, stride, 1, bias=False, norm=get_norm("BN", oup))
    weight_init.c2_msra_fill(conv)
    return nn.Sequential(
        conv,
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride, width_mult=1.0):
    inp = 3 if inp == 3 else scale_chan(inp, width_mult)
    oup = scale_chan(oup, width_mult)
    conv1 = Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False, norm=get_norm("BN", inp))
    conv2 = Conv2d(inp, oup, 1, 1, 0, bias=False, norm=get_norm("BN", oup))
    weight_init.c2_msra_fill(conv1)
    weight_init.c2_msra_fill(conv2)
    return nn.Sequential(
        conv1,
        nn.ReLU(inplace=True),
        conv2,
        nn.ReLU(inplace=True),
    )

class MobileNetV1(Backbone):
    def __init__(self, width_mult=1.0, min_chan=16):
        super(MobileNetV1, self).__init__()
        self._out_feature_strides = {
            "s1": 2,
            "s2": 4,
            "s3": 8,
            "s4": 16,
            "s5": 32,
        }
        self._out_feature_channels = {
            "s1": scale_chan(64, width_mult),
            "s2": scale_chan(128, width_mult),
            "s3": scale_chan(256, width_mult),
            "s4": scale_chan(512, width_mult),
            "s5": scale_chan(1024, width_mult),
        }
        self._out_features = ["linear"]
        self.stage1 = nn.Sequential(
            conv_bn(3, 32, 2, width_mult), 
            conv_dw(32, 64, 1, width_mult),
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2, width_mult),
            conv_dw(128, 128, 1, width_mult),
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2, width_mult),
            conv_dw(256, 256, 1, width_mult),
        )
        self.stage4 = nn.Sequential(
            conv_dw(256, 512, 2, width_mult),
            conv_dw(512, 512, 1, width_mult),
            conv_dw(512, 512, 1, width_mult),
            conv_dw(512, 512, 1, width_mult),
            conv_dw(512, 512, 1, width_mult),
            conv_dw(512, 512, 1, width_mult),
        )
        self.stage5 = nn.Sequential(
            conv_dw(512, 1024, 2, width_mult),
            conv_dw(1024, 1024, 1, width_mult),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(scale_chan(1024, width_mult), 1000)
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, x):
        outputs = {}
        x = self.stage1(x)
        outputs["s1"] = x
        x = self.stage2(x)
        outputs["s2"] = x
        x = self.stage3(x)
        outputs["s3"] = x
        x = self.stage4(x)
        outputs["s4"] = x
        x = self.stage5(x)
        outputs["s5"] = x
        x = self.avgpool(x)
        outputs["avg"] = x
        x = nn.Flatten()(x)
        x = self.linear(x)
        outputs["linear"] = x

        return {k:outputs[k] for k in self._out_features}

@BACKBONE_REGISTRY.register()
def build_mobilenetv1_backbone(cfg, input_shape):
    return MobileNetV1()
