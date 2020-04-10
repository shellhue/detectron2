import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from detectron2.modeling.backbone.resnet import ResNetBlockBase

from detectron2.modeling.backbone import Backbone
from detectron2.modeling import BACKBONE_REGISTRY

__all__ = [
    "Conv2dBNLeakyReLU",
    "DarknetBottleneckBlock",
    "DarknetBasicStem",
    "darknet_make_stage",
    "build_darknet_backbone",
]


class Conv2dBNLeakyReLU(ResNetBlockBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 *,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 norm="BN"):
        super(Conv2dBNLeakyReLU, self).__init__(
            in_channels, out_channels, stride)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = get_norm(norm, out_channels)

        # parameter init
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu_(x, negative_slope=0.1)
        return x


class DarknetBottleneckBlock(ResNetBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        norm="BN",
    ):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2dBNLeakyReLU(
            in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, norm=norm)

        self.conv2 = Conv2dBNLeakyReLU(
            bottleneck_channels, out_channels, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


def darknet_make_stage(block_class,
                       num_blocks,
                       in_channels,
                       out_channels,
                       *,
                       bottleneck_channels,
                       norm="BN",):
    """
    Create a resnet stage by creating many blocks.
    Args:
        block_class (class): a subclass of ResNetBlockBase
        num_blocks (int):
        kwargs: other arguments passed to the block constructor.

    Returns:
        list[nn.Module]: a list of block module.
    """
    blocks = [
        Conv2dBNLeakyReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm=norm)
    ]
    for _ in range(num_blocks):
        blocks.append(block_class(
            in_channels=out_channels,
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            norm=norm))
    return blocks


class DarknetBasicStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, norm="BN"):
        """
        Args:
            norm (str or callable): a callable that takes the number of
                channels and return a `nn.Module`, or a pre-defined string
                (one of {"FrozenBN", "BN", "GN"}).
        """
        super().__init__()
        self.conv1 = Conv2dBNLeakyReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=norm
        )

    def forward(self, x):
        return self.conv1(x)

    @property
    def out_channels(self):
        return self.conv1.out_channels

    @property
    def stride(self):
        return 1


class Darknet(Backbone):
    def __init__(self, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[ResNetBlock]]): several (typically 4) stages,
                each contains multiple :class:`ResNetBlockBase`.
            num_classes (None or int): if None, will not perform classification.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super(Darknet, self).__init__()
        assert ("linear" in out_features and num_classes is not None) or ("linear" not in out_features and num_classes in [None, 0]), "linear output needs num_classes not None"
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            for block in blocks:
                assert isinstance(block, ResNetBlockBase) or isinstance(
                    block, Conv2dBNLeakyReLU), block
                curr_channels = block.out_channels
            stage = nn.Sequential(*blocks)
            name = "s" + str(i + 1)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = blocks[-1].out_channels

        if num_classes not in [None, 0]:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)
            # self.linear = nn.Conv2d(
            #     curr_channels,
            #     num_classes,
            #     kernel_size=1,
            #     stride=1
            # )
            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            # weight_init.c2_msra_fill(self.linear)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(
                ", ".join(children))
        
    def forward(self, x, targets=None):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes not in [None, 0]:
            x = self.avgpool(x)
            x = nn.Flatten()(x)
            x = self.linear(x)
            
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def get_conv_bn_modules(self, include_linear=False):
        """
        for weight convert from original yolo weights file
        """
        modules = []
        count = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                modules.append(module)
            if isinstance(module, nn.BatchNorm2d):
                modules.append(module)
            if include_linear and isinstance(module, nn.Linear):
                modules.append(module)
        return modules


@BACKBONE_REGISTRY.register()
def build_darknet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.DAKRNET.NORM
    stem = DarknetBasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.DAKRNET.STEM_OUT_CHANNELS,
        norm=norm,
    )
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    if freeze_at >= 1:
        for p in stem.parameters():
            p.requires_grad = False
        stem = FrozenBatchNorm2d.convert_frozen_batchnorm(stem)

    # fmt: off
    # OUT_FEATURES: ["s1", "s2", "s3", "s4", "s5", "gap", "linear"]
    out_features = cfg.MODEL.DAKRNET.OUT_FEATURES
    depth = cfg.MODEL.DAKRNET.DEPTH
    in_channels = cfg.MODEL.DAKRNET.STEM_OUT_CHANNELS

    num_blocks_per_stage = {53: [1, 2, 8, 8, 4]}[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    if "linear" in out_features:
        max_stage_idx = 5
    else:
        out_stage_idx = [{"s1": 1, "s2": 2, "s3": 3, "s4": 4, "s5": 5}[f]
                        for f in out_features]
        max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(1, max_stage_idx + 1)):
        stage_kargs = {
            "block_class": DarknetBottleneckBlock,
            "num_blocks": num_blocks_per_stage[idx],
            "in_channels": in_channels,
            "bottleneck_channels": in_channels,
            "out_channels": in_channels * 2,
            "norm": norm,
        }
        blocks = darknet_make_stage(**stage_kargs)
        in_channels *= 2

        if freeze_at >= stage_idx:
            for block in blocks:
                block.freeze()
        stages.append(blocks)
    return Darknet(stem, stages, out_features=out_features, num_classes=cfg.MODEL.BACKBONE.NUM_CLASSES)
