# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import math
from typing import List
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.registry import Registry
from detectron2.modeling.anchor_generator import BufferList, ANCHOR_GENERATOR_REGISTRY

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

@ANCHOR_GENERATOR_REGISTRY.register()
class YoloAnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        sizes = cfg.MODEL.ANCHOR_GENERATOR.SIZES
        # aspect_ratios = cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS
        self.strides = [x.stride for x in input_shape]
        # fmt: on
        """
        sizes (list[list[(int,int)]]): sizes[i] is the list of anchor sizes to use
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

        self.num_features = len(self.strides)
        self.cell_anchors = self._calculate_anchors(sizes)

    def _calculate_anchors(self, sizes):
        # If one size (or aspect ratio) is specified and there are multiple feature
        # maps, then we "broadcast" anchors of that single size (or aspect ratio)
        # over all feature maps.

        cell_anchors = [
            self.generate_cell_anchors(s).float() for s in sizes
        ]

        return BufferList(cell_anchors)

    @property
    def box_dim(self):
        """
        Returns:
            int: the dimension of each anchor box.
        """
        return 4

    @property
    def num_cell_anchors(self):
        """
        Returns:
            list[int]: Each int is the number of anchors at every pixel
                location, on that feature map.
                For example, if at every pixel we use anchors of 3 aspect
                ratios and 5 sizes, the number of anchors is 15.
                (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

                In standard RPN models, `num_cell_anchors` on every feature map is the same.
        """
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):
            shift_x, shift_y = _create_grid_offsets(size, 1, base_anchors.device)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4) + stride / 2.0)

        return anchors

    def generate_cell_anchors(self, sizes=[[32.0, 64.0], [128.0, 256.0]]):
        """
        Generate a tensor storing anchor boxes, which are continuous geometric rectangles
        centered on one feature map point sample. We can later build the set of anchors
        for the entire feature map by tiling these tensors; see `meth:grid_anchors`.

        Args:
            sizes (list[list(float, float)]): Absolute size of the anchors in the units of the input
                image (the input received by the network, after undergoing necessary scaling).
                The absolute size is given as the side length of a box.

        Returns:
            Tensor of shape (len(sizes), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            w, h = size
            x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
            anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate anchors.

        Returns:
            list[list[Boxes]]: a list of #image elements. Each is a list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)

        anchors_in_image = []
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            boxes = Boxes(anchors_per_feature_map)
            anchors_in_image.append(boxes)

        anchors = [copy.deepcopy(anchors_in_image) for _ in range(num_images)]
        return anchors
