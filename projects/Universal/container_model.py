import sys
sys.path.append('/home/huangzeyu/tmp/yolov3')

import math
import torch
import numpy as np
from torch import nn
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

# import for specific model
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
        # x is the output of dataloader

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
        imgs = imgs.float() / 255.0

        return imgs
