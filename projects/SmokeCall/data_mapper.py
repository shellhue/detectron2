# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import random

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["SmokeDatasetMapper"]


class SmokeDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        self.img_format = cfg.INPUT.FORMAT

        self.is_train = is_train

        if self.is_train:
            self.tfs = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.tfs = transforms.Compose([
                transforms.Resize(20),
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)

        img = Image.open(dataset_dict["file_name"])
        if not img.mode == "RGB":
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg
        if self.is_train:
            self.tfs = transforms.Compose([
                transforms.Resize(random.randint(20, 256)), # 由于头像分辨率在实际中变化较大，这里模拟多分辨率输入的情形
                transforms.RandomResizedCrop(224), # 再把头像resize到固定分辨率
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        img = self.tfs(img)
        dataset_dict["image"] = img

        return dataset_dict
