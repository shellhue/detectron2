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


class HNet(Backbone):
    def __init__(self):
        super(HNet, self).__init__()
        self._out_feature_strides = {"stem": 1, "s1": 1}
        self._out_feature_channels = {"stem": 64, "s1": 64}
        self._out_features = ["s1"]
        blocks = []
        self.conv1 = Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm("BN", 64),
        )
        weight_init.c2_msra_fill(self.conv1)
        for i in range(16):
            blocks.append(BottleneckBlock(64, 64, bottleneck_channels=32))
        self.s1 = nn.Sequential(*blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 1000)
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = F.relu_(x)
        x = self.s1(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)

        x = self.linear(x)
        outputs["linear"] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_hnet_backbone(cfg, input_shape):
    return HNet()
