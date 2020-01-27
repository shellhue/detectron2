import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
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

import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        Conv2d(inp, oup, 1, 1, 0, bias=False, norm=get_norm("BN", oup)),
        nn.ReLU6(inplace=True)
    )

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False, norm=get_norm("BN", hidden_dim)),
                nn.ReLU6(inplace=True),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, norm=get_norm("BN", oup)),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                Conv2d(inp, hidden_dim, 1, 1, 0, bias=False, norm=get_norm("BN", hidden_dim)),
                nn.ReLU6(inplace=True),
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False, norm=get_norm("BN", hidden_dim)),
                nn.ReLU6(inplace=True),
                # pw-linear
                Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, norm=get_norm("BN", oup)),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def get_modules(setting, input_channel, block, width_mult=1.):
    t, c, n, s = setting
    output_channel = make_divisible(c * width_mult) if t > 1 else c
    features = []
    for i in range(n):
        if i == 0:
            features.append(block(input_channel, output_channel, s, expand_ratio=t))
        else:
            features.append(block(input_channel, output_channel, 1, expand_ratio=t))
        input_channel = output_channel
    return features, output_channel

class MobileNetV2(Backbone):
    def __init__(self, n_class=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self._out_feature_strides = {
            "s1": 2,
            "s2": 4,
            "s3": 8,
            "s4": 16,
            "s5": 32,
        }
        self._out_feature_channels = {}
        self._out_features = ["linear"]

        block = InvertedResidual
        
        last_channel = 1280
        # interverted_residual_setting = [
        #     # t, c, n, s
        #     [1, 16, 1, 1],
        #     [6, 24, 2, 2],
        #     [6, 32, 3, 2],
        #     [6, 64, 4, 2],
        #     [6, 96, 3, 1],
        #     [6, 160, 3, 2],
        #     [6, 320, 1, 1],
        # ]

        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        # stage1
        features = []
        input_channel = 32
        features.append(conv_bn(3, input_channel, 2))
        fs, input_channel = get_modules([1, 16, 1, 1], input_channel, block, width_mult=width_mult)
        features.extend(fs)
        self.stage1 = nn.Sequential(*features)
        self._out_feature_channels["s1"] = input_channel

        # stage2
        features = []
        fs, input_channel = get_modules([6, 24, 2, 2], input_channel, block, width_mult=width_mult)
        features.extend(fs)
        self.stage2 = nn.Sequential(*features)
        self._out_feature_channels["s2"] = input_channel

        # stage3
        features = []
        fs, input_channel = get_modules([6, 32, 3, 2], input_channel, block, width_mult=width_mult)
        features.extend(fs)
        self.stage3 = nn.Sequential(*features)
        self._out_feature_channels["s3"] = input_channel

        # stage4
        features = []
        fs, input_channel = get_modules([6, 64, 4, 2], input_channel, block, width_mult=width_mult)
        features.extend(fs)
        fs, input_channel = get_modules([6, 96, 3, 1], input_channel, block, width_mult=width_mult)
        features.extend(fs)
        self.stage4 = nn.Sequential(*features)
        self._out_feature_channels["s4"] = input_channel

        # stage5
        features = []
        fs, input_channel = get_modules([6, 160, 3, 2], input_channel, block, width_mult=width_mult)
        features.extend(fs)
        fs, input_channel = get_modules([6, 320, 1, 1], input_channel, block, width_mult=width_mult)
        features.extend(fs)
        features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.stage5 = nn.Sequential(*features)
        self._out_feature_channels["s5"] = self.last_channel

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_msra_fill(m)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)

    return model

@BACKBONE_REGISTRY.register()
def build_mobilenetv2_backbone(cfg, input_shape):
    return MobileNetV2(width_mult=1)