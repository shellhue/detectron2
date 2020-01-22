
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_backbone_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg
    
    # Darknet
    _C.MODEL.DAKRNET = CN()
    _C.MODEL.DAKRNET.NORM = "BN"
    _C.MODEL.DAKRNET.NUM_CLASSES = 1000
    _C.MODEL.DAKRNET.STEM_OUT_CHANNELS = 32
    _C.MODEL.DAKRNET.OUT_FEATURES = ["linear"]
    _C.MODEL.DAKRNET.DEPTH = 53
