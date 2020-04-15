import sys
# sys.path.append('/home/huangzeyu/tmp/EfficientDet.Pytorch')
sys.path.append('/home/huangzeyu/tmp/Yet-Another-EfficientDet-Pytorch')

import math
import torch
import numpy as np
from torch import nn

# import for specific model
from detectron2_bridger import build_efficient_det_model, compute_loss

class ProxyModel(nn.Module):
    def __init__(self, cfg=None):
        super(ProxyModel, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.back_model = build_efficient_det_model(num_class=80, compound_coef=0)
        self.to(self.device)
    
    def forward(self, x):
        # x is the output of dataloader
        if self.training:
            return compute_loss(x, self.back_model, self.device)
        else:
            raise ValueError("proxy model can just be used to train")