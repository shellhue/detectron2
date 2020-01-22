import copy
import itertools
import logging
import numpy as np
import torch.utils.data

from detectron2.utils.comm import get_world_size

from detectron2.data import samplers
# from detectron2.data.build import build_batch_data_sampler
from detectron2.data.build import trivial_batch_collator
from detectron2.data.build import worker_init_reset_seed
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset

from .register_imagenet import register_all_imagenet
from .data_mapper import ClassificationDatasetMapper


def get_classification_dataset_dicts(dataset_names):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (list[str]): a list of dataset names
    """
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name)
                     for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    return dataset_dicts

def build_classification_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:
      * Map each metadata dict into another format to be consumed by the model.
      * Batch them by simply putting dicts into a list.
    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        an infinite iterator of training data
    """
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

    dataset_dicts = get_classification_dataset_dicts(cfg.DATASETS.TRAIN)
    dataset = DatasetFromList(dataset_dicts, copy=False)

    # Bin edges for batching images with similar aspect ratios. If ASPECT_RATIO_GROUPING
    # is enabled, we define two bins with an edge at height / width = 1.
    if mapper is None:
        mapper = ClassificationDatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
    #     data_loader = torch.utils.data.DataLoader(
    #         dataset,
    #         sampler=sampler,
    #         num_workers=cfg.DATALOADER.NUM_WORKERS,
    #         batch_sampler=None,
    #         collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
    #         worker_init_fn=worker_init_reset_seed,
    #     )  # yield individual mapped dict
    #     data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    # else:
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_worker, drop_last=True
    )
    # drop_last so the batch always have the same size
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )

    return data_loader

# def build_classification_train_loader(cfg, mapper=None):
#     """
#     A data loader is created by the following steps:

#     1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
#     2. Start workers to work on the dicts. Each worker will:
    #   * Map each metadata dict into another format to be consumed by the model.
    #   * Batch them by simply putting dicts into a list.
    # The batched ``list[mapped_dict]`` is what this dataloader will return.

    # Args:
    #     cfg (CfgNode): the config
    #     mapper (callable): a callable which takes a sample (dict) from dataset and
    #         returns the format to be consumed by the model.
    #         By default it will be `DatasetMapper(cfg, True)`.

    # Returns:
    #     a torch DataLoader object
    # """
    # num_workers = get_world_size()
    # images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    # assert (
    #     images_per_batch % num_workers == 0
    # ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
    #     images_per_batch, num_workers
    # )
    # assert (
    #     images_per_batch >= num_workers
    # ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
    #     images_per_batch, num_workers
    # )
    # images_per_worker = images_per_batch // num_workers

    # dataset_dicts = get_classification_dataset_dicts(cfg.DATASETS.TRAIN)
    # dataset = DatasetFromList(dataset_dicts, copy=False)

    # # Bin edges for batching images with similar aspect ratios. If ASPECT_RATIO_GROUPING
    # # is enabled, we define two bins with an edge at height / width = 1.
    # if mapper is None:
    #     mapper = ClassificationDatasetMapper(cfg, True)
    # dataset = MapDataset(dataset, mapper)

    # sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    # logger = logging.getLogger(__name__)
    # logger.info("Using training sampler {}".format(sampler_name))
    # if sampler_name == "TrainingSampler":
    #     sampler = samplers.TrainingSampler(len(dataset))
    # elif sampler_name == "RepeatFactorTrainingSampler":
    #     sampler = samplers.RepeatFactorTrainingSampler(
    #         dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
    #     )
    # else:
    #     raise ValueError("Unknown training sampler: {}".format(sampler_name))
    # batch_sampler = build_batch_data_sampler(
    #     sampler, images_per_worker
    # )

    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     num_workers=cfg.DATALOADER.NUM_WORKERS,
    #     batch_sampler=batch_sampler,
    #     collate_fn=trivial_batch_collator,
    #     worker_init_fn=worker_init_reset_seed,
    # )
    # return data_loader


def build_classification_test_loader(cfg, dataset_name, mapper=None):
    """
    Similar to `build_detection_train_loader`.
    But this function uses the given `dataset_name` argument (instead of the names in cfg),
    and uses batch size 1.

    Args:
        cfg: a detectron2 CfgNode
        dataset_name (str): a name of the dataset that's available in the DatasetCatalog
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           By default it will be `DatasetMapper(cfg, False)`.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    dataset_dicts = get_classification_dataset_dicts(cfg.DATASETS.TEST)
    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = ClassificationDatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader
