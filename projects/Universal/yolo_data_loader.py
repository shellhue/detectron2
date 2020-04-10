import os
import sys
sys.path.append('.')
sys.path.append('/home/huangzeyu/tmp/yolov3')

import torch
from detectron2.data import samplers
from utils.datasets import *
from utils.utils import *
from detectron2.utils.comm import get_world_size

def build_yolo_detection_train_loader(cfg, mapper=None):
    hyp = {
        'giou': 3.54, 
        'cls': 37.4, 
        'cls_pw': 1.0, 
        'obj': 64.3, 
        'obj_pw': 1.0, 
        'iou_t': 0.225, 
        'lr0': 0.01, 
        'lrf': -4.0, 
        'momentum': 0.937, 
        'weight_decay': 0.000484, 
        'fl_gamma': 0.0, 
        'hsv_h': 0.0138, 
        'hsv_s': 0.678, 
        'hsv_v': 0.36, 
        'degrees': 0.0,
        'translate': 0.0, 
        'scale': 0.0,
        'shear': 0.0
    }
    train_path = "/home/huangzeyu/tmp/yolov3/data/coco/trainvalno5k.txt"
    img_size = 416
    
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
    dataset = LoadImagesAndLabels(train_path, img_size, images_per_worker,
                                  augment=True,
                                  hyp=hyp,
                                  rect=False,
                                  cache_images=False,
                                  single_cls=False)
    sampler = samplers.TrainingSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_worker, drop_last=True
    )
    # Dataloader
    images_per_worker = min(images_per_worker, len(dataset))
    nw = min([os.cpu_count(), images_per_worker if images_per_worker > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=nw,
                                             collate_fn=dataset.collate_fn)

    return dataloader


def build_yolo_detection_test_loader(cfg, mapper=None):
    hyp = {
        'giou': 3.54, 
        'cls': 37.4, 
        'cls_pw': 1.0, 
        'obj': 64.3, 
        'obj_pw': 1.0, 
        'iou_t': 0.225, 
        'lr0': 0.01, 
        'lrf': -4.0, 
        'momentum': 0.937, 
        'weight_decay': 0.000484, 
        'fl_gamma': 0.0, 
        'hsv_h': 0.0138, 
        'hsv_s': 0.678, 
        'hsv_v': 0.36, 
        'degrees': 0.0,
        'translate': 0.0, 
        'scale': 0.0,
        'shear': 0.0
    }
    train_path = "/home/huangzeyu/tmp/yolov3/data/coco/trainvalno5k.txt"
    img_size = 416
    
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
    dataset = LoadImagesAndLabels(
        "/home/huangzeyu/tmp/yolov3/data/coco/5k.txt", 
        416, 
        images_per_worker,
        hyp=hyp,
        rect=True,
        cache_images=False,
        single_cls=False)

    # Dataloader
    images_per_worker = min(images_per_worker, len(dataset))
    nw = min([os.cpu_count(), images_per_worker if images_per_worker > 1 else 0, 8])  # number of workers
    sampler = samplers.InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, 1, drop_last=False)

    test_loader = torch.utils.data.DataLoader(dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=nw,
                                             collate_fn=dataset.collate_fn)

    return test_loader