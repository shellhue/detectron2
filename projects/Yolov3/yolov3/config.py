
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_yolov3_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.GRID_GENERATOR = CN()
    _C.MODEL.GRID_GENERATOR.NAME = "DefaultGridGenerator"
    _C.MODEL.STRIDE_GENERATOR = CN()
    _C.MODEL.STRIDE_GENERATOR.NAME = "DefaultStrideGenerator"

    _C.MODEL.DAKRNET = CN()
    _C.MODEL.DAKRNET.NORM = "BN"
    _C.MODEL.DAKRNET.NUM_CLASSES = 1000
    _C.MODEL.DAKRNET.STEM_OUT_CHANNELS = 32
    _C.MODEL.DAKRNET.OUT_FEATURES = ["s3", "s4", "s5"]
    _C.MODEL.DAKRNET.DEPTH = 53

    _C.MODEL.DarknetFPN = CN()
    _C.MODEL.DarknetFPN.IN_FEATURES = ["s3", "s4", "s5"]
    _C.MODEL.DarknetFPN.OUT_CHANNELS = [128, 256, 512]

    _C.MODEL.YOLOV3 = CN()
    _C.MODEL.YOLOV3.NORM = "BN"
    _C.MODEL.YOLOV3.NUM_CLASSES = 80
    _C.MODEL.YOLOV3.IN_FEATURES = ["p3", "p4", "p5"]
    _C.MODEL.YOLOV3.HEAD = CN()
    _C.MODEL.YOLOV3.HEAD.OUT_CHANNELS = [256, 512, 1024]
