# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Yolov3 Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import sys
sys.path.append('.')
sys.path.append('/home/huangzeyu/tmp/yolov3')

import logging
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, verify_results

from data_loader import build_yolo_detection_train_loader
from data_loader import build_yolo_detection_test_loader


from proxy_model import ProxyModel
class Trainer(DefaultTrainer):
    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     """
    #     Create evaluator(s) for a given dataset.
    #     This uses the special metadata "evaluator_type" associated with each builtin dataset.
    #     For your own dataset, you can simply create an evaluator manually in your
    #     script and do not have to worry about the hacky if-else logic here.
    #     """
    #     if output_folder is None:
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #     evaluator_list = []
    #     evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    #     if evaluator_type in ["classification"]:
    #         evaluator_list.append(
    #             ClassificationEvaluator(dataset_name)
    #         )
    #     if len(evaluator_list) == 1:
    #         return evaluator_list[0]
    #     return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        loader = build_yolo_detection_train_loader(cfg)
        return loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_yolo_detection_test_loader(cfg)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """

        model = ProxyModel(cfg=cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        model.train()
        return model

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_backbone_config(cfg)
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
