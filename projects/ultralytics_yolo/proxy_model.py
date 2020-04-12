import sys
sys.path.append('/home/huangzeyu/tmp/yolov3')

import math
import torch
import numpy as np
from torch import nn
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import random

# import for specific model
from train import create_model as create_yolov3_model
from utils.datasets import *
from utils.utils import *

class ProxyModel(nn.Module):
    def __init__(self, cfg=None):
        super(ProxyModel, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.back_model = create_yolov3_model(weights="/home/huangzeyu/tmp/yolov3/weights/darknet53.conv.74")
        self.to(self.device)

        img_size = 416
        self.img_sz_min = round(img_size / 32 / 1.5)
        self.img_sz_max = round(img_size / 32 * 1.5)
    
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
        
        img_size = random.randrange(self.img_sz_min, self.img_sz_max + 1) * 32
        sf = img_size / max(imgs.shape[2:])  # scale factor
        if sf != 1:
            ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
            imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        return imgs
