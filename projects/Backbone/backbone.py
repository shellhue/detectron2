import math
import torch
import numpy as np
from torch import nn
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
import nets

@META_ARCH_REGISTRY.register()
class ClassificationBackbone(nn.Module):
    def __init__(self, cfg=None):
        super(ClassificationBackbone, self).__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_classes = 1000

        self.backbone = build_backbone(cfg)


        self.to(self.device)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        labels = self.get_gt_labels(batched_inputs)
        features = self.backbone(images.tensor)["linear"]

        if self.training:
            losses = {
                "classes": F.cross_entropy(features, labels.long())
            }
            return losses
        else:
            return F.softmax(features, dim=-1)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]

        images = ImageList.from_tensors(images)
        return images
    
    def get_gt_labels(self, batched_inputs):
        labels = [x["id"] for x in batched_inputs]
        labels = torch.tensor(labels).int().to(self.device)
        return labels