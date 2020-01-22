import math
import torch
import numpy as np
from torch import nn
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

from projects.Backbone.nets.darknet53 import Conv2dBNLeakyReLU


class Yolov3BoxPredictor(nn.Module):
    """
    Top down means direction from output to input (from low to high resolution).
    Bottom up means direction from input to output (from high to low resolution).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_classes=80,
                 num_anchors_per_cell=3,
                 norm="BN"):
        super(Yolov3BoxPredictor, self).__init__()
        self.conv1 = Conv2dBNLeakyReLU(in_channels, out_channels,
                              kernel_size=3, stride=1, padding=1, norm=norm)
        self.conv2 = nn.Conv2d(out_channels, num_anchors_per_cell * (num_classes + 5),
                      kernel_size=1, stride=1, padding=0)
        torch.nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        torch.nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x

class Yolov3Head(nn.Module):
    def __init__(self,
                 in_features,
                 in_channels,
                 out_channels,
                 num_classes=80,
                 num_anchors_per_cell=3,
                 norm="BN"):
        super(Yolov3Head, self).__init__()
        assert len(in_channels) == len(out_channels), "in channels length should be equal to out channels."
        self.heads = []
        for i in range(len(in_features)):
            h = Yolov3BoxPredictor(
                in_channels[i],
                out_channels[i],
                num_classes=num_classes, 
                num_anchors_per_cell=num_anchors_per_cell, 
                norm=norm)
            self.add_module(in_features[i], h)
            self.heads.append(h)

    def forward(self, x):
        results = []
        for features, head in zip(x, self.heads):
            results.append(head(features))
        return results
    
    def get_conv_bn_modules(self):
        """
        for weight convert from original yolo weights file
        """
        modules = []
        for head in self.heads:
            modules_i = []
            for name, module in head.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    modules_i.append(module)
            modules.append(modules_i)
        return modules[::-1]