# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import math
from typing import List
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.registry import Registry

STRIDE_GENERATOR_REGISTRY = Registry("STRIDE_GENERATOR")
"""
Registry for modules that creates object detection anchors for feature maps.
"""


@STRIDE_GENERATOR_REGISTRY.register()
class DefaultStrideGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        self.strides  = [x.stride for x in input_shape]
        self.num_anchors = [len(s) for s in cfg.MODEL.ANCHOR_GENERATOR.SIZES]
        self.device = cfg.MODEL.DEVICE
        """
        sizes (list[list[int]]): sizes[i] is the list of anchor sizes to use
            for the i-th feature map. If len(sizes) == 1, then the same list of
            anchor sizes, given by sizes[0], is used for all feature maps. Anchor
            sizes are given in absolute lengths in units of the input image;
            they do not dynamically scale if the input image size changes.
        aspect_ratios (list[list[float]]): aspect_ratios[i] is the list of
            anchor aspect ratios to use for the i-th feature map. If
            len(aspect_ratios) == 1, then the same list of anchor aspect ratios,
            given by aspect_ratios[0], is used for all feature maps.
        strides (list[int]): stride of each input feature.
        """

    def _create_grid(self, grid_sizes, device):
        grids = []
        for size, num_anchor, stride in zip(grid_sizes, self.num_anchors, self.strides):
            h, w = size
            stride_grid = torch.ones((h, w, num_anchor), device=device, dtype=torch.float) * stride

            grids.append(stride_grid.reshape(-1))

        return grids


    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[list[Boxes]]: a list of #image elements. Each is a list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
        """
        num_images = len(features[0])
        device = features[0].device
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        grid_over_all_feature_maps = self._create_grid(grid_sizes, device)

        grids = [copy.deepcopy(grid_over_all_feature_maps) for _ in range(num_images)]
        return grids


def build_stride_generator(cfg, input_shape):
    """
    Built an stride generator from `cfg.MODEL.ANCHOR_GENERATOR.NAME`.
    """
    stride_generator = cfg.MODEL.STRIDE_GENERATOR.NAME
    return STRIDE_GENERATOR_REGISTRY.get(stride_generator)(cfg, input_shape)
