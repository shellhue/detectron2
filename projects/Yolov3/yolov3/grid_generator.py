# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import math
from typing import List
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.registry import Registry

GRID_GENERATOR_REGISTRY = Registry("GRID_GENERATOR")
"""
Registry for modules that creates object detection anchors for feature maps.
"""


def _create_grid_offsets(size, stride, device):
    grid_height, grid_width = size
    shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(
        0, grid_height * stride, step=stride, dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


@GRID_GENERATOR_REGISTRY.register()
class DefaultGridGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        self.num_anchors = [len(s) for s in cfg.MODEL.ANCHOR_GENERATOR.SIZES]
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
        for size, num_anchor in zip(grid_sizes, self.num_anchors):
            shift_x, shift_y = _create_grid_offsets(size, 1, device)
            shifts = torch.stack((shift_x, shift_y), dim=1)

            grids.append(shifts.view(-1, 1, 2).repeat(1, num_anchor, 1).reshape(-1, 2))

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


def build_grid_generator(cfg, input_shape):
    """
    Built an grid generator from `cfg.MODEL.GRID_GENERATOR.NAME`.
    """
    grid_generator = cfg.MODEL.GRID_GENERATOR.NAME
    return GRID_GENERATOR_REGISTRY.get(grid_generator)(cfg, input_shape)
