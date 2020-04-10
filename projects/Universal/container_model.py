import sys
sys.path.append('/home/huangzeyu/tmp/yolov3')

import math
import torch
import numpy as np
from torch import nn
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from train import create_model as create_yolov3_model
from utils.datasets import *
from utils.utils import *

class ContainerModel(nn.Module):
    def __init__(self, cfg=None):
        super(ContainerModel, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.back_model = create_yolov3_model(weights="/home/huangzeyu/tmp/yolov3/weights/darknet53.conv.74")
        self.to(self.device)

    def forward(self, x):
        imgs, labels, paths, _ = x
        imgs = self.preprocess_image(imgs.to(self.device))
        labels = labels.to(self.device)

        if self.training:
            pred = self.back_model(imgs)
            l, loss_items = compute_loss(pred, labels, self.back_model)
            losses = {
                "loss": l
            }
            return losses
        else:
            return self.back_model(imgs)

    def preprocess_image(self, imgs):
        """
        Normalize, pad and batch the input images.
        """
        imgs = imgs.float() / 255.0

        # # Multi-Scale training
        # if True:
        #     if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
        #         img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
        #     sf = img_size / max(imgs.shape[2:])  # scale factor
        #     if sf != 1:
        #         ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
        #         imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
        return imgs
