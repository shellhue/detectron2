import os
import sys
sys.path.append('.')

from train_net import Trainer
from detectron2.config import get_cfg
from projects.Backbone.config import add_backbone_config
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
import torch
    
def create_model(config_file, weights):
    cfg = get_cfg()
    add_backbone_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()


    model = Trainer.build_model(cfg)

    state_dict = torch.load(weights)["model"]
    model.load_state_dict(state_dict)

    return model
    
