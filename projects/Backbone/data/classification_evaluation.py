# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import os
from collections import OrderedDict
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.evaluation.evaluator import DatasetEvaluator


class ClassificationEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._class_names = meta.thing_classes
        self._class_ids = meta.class_ids
        self._class_id_to_name = meta.class_id_to_name
        # self._class_id_to_short_name = meta.class_id_to_short_name
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predictions = {}
        self._gts = {}

    def reset(self):
        self._predictions = {}
        self._gt = {}

    def process(self, inputs, outputs):
        probs, idxs = torch.sort(outputs, dim=-1, descending=True)
        top1 = idxs[..., 0].cpu().numpy().tolist()
        top5 = idxs[..., :5].cpu().numpy().tolist()
        for i, input_i in enumerate(inputs):
            top5_i = top5[i]
            file_name = input_i["file_name"]
            gt_label = input_i["id"]
            self._gts[file_name] = gt_label
            self._predictions[file_name] = top5_i

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        all_gts = comm.gather(self._gts, dst=0)
        predits = {}
        gts = {}
        for predits_i, gts_i in zip(all_predictions, all_gts):
            predits.update(predits_i)
            gts.update(gts_i)
        assert len(gts) == len(
            predits), "label count should be equal to predict count."
        if not comm.is_main_process():
            return

        self._logger.info(
            "Evaluating {} using classification metric. ".format(
                self._dataset_name
            )
        )

        top1_tp = 0
        top5_tp = 0
        total_sample_num = len(predits)

        for i, predict in predits.items():
            gt_label = gts[i]
            if gt_label == predict[0]:
                top1_tp += 1
            if gt_label in predict:
                top5_tp += 1

        ret = OrderedDict()
        ret["cls"] = {
            "top1": top1_tp / total_sample_num,
            "top5": top5_tp / total_sample_num
        }

        return ret
