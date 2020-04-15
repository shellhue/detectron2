# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Yolov3 Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import sys
import torch
from torch import nn
sys.path.append('.')

import logging
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.utils.comm import get_world_size
from detectron2.data import samplers

from proxy_model import ProxyModel

# import for specific model
sys.path.append('/home/huangzeyu/tmp/Yet-Another-EfficientDet-Pytorch')
from detectron2_bridger import build_train_dataset, build_val_dataset

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        num_workers = get_world_size()
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_workers == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
            images_per_batch, num_workers
        )
        assert (
            images_per_batch >= num_workers
        ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
            images_per_batch, num_workers
        )
        images_per_worker = images_per_batch // num_workers

        # Dataset
        dataset, collactor = build_train_dataset(0)

        # Sampler
        sampler = samplers.TrainingSampler(len(dataset))
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )
        # Dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_sampler=batch_sampler,
                                                num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                collate_fn=collactor)
        return dataloader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        num_workers = get_world_size()
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_workers == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
            images_per_batch, num_workers
        )
        assert (
            images_per_batch >= num_workers
        ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
            images_per_batch, num_workers
        )
        images_per_worker = images_per_batch // num_workers

        # Dataset
        dataset, collactor = build_val_dataset(0)

        # Dataloader
        sampler = samplers.InferenceSampler(len(dataset))
        # Always use 1 image per worker during inference since this is the
        # standard when reporting inference time in papers.
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, 1, drop_last=False)

        test_loader = torch.utils.data.DataLoader(dataset,
                                                batch_sampler=batch_sampler,
                                                num_workers=cfg.DATALOADER.NUM_WORKERS,
                                                collate_fn=collactor)
        return test_loader

    @classmethod
    def build_model(cls, cfg):
        model = ProxyModel(cfg=cfg)
        logger = logging.getLogger("detectron2")
        logger.info("Model:\n{}".format(model))
        model.train()
        return model

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
