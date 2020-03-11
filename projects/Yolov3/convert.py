import torch.nn as nn
import numpy as np
import torch

import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from yolov3 import add_yolov3_config

def load_darknet_weights(weights, modules):
    with open(weights, 'rb') as f:
        # (int32) version info: major, minor, revision
        version = np.fromfile(f, dtype=np.int32, count=3)
        # (int64) number of images seen during training
        seen = np.fromfile(f, dtype=np.int64, count=1)
        # the rest are weights
        weights = np.fromfile(f, dtype=np.float32)
        print(version, seen)
        print(weights.shape)
        

    ptr = 0
    paired_modules = []
    param_count = 0
    for i, module in enumerate(modules):
        if isinstance(module, nn.Conv2d):
            if not module.bias is None:
                paired_modules.append([module])
                param_count += module.weight.numel()
                param_count += module.bias.numel()
            else:
                paired_modules.append([module, modules[i+1]])
                param_count += module.weight.numel()
                param_count += modules[i+1].bias.numel() * 4
    print("param_count:", param_count)
    for conv_bn_modules in paired_modules:
        conv = conv_bn_modules[0]
        bn = conv_bn_modules[1] if len(conv_bn_modules) == 2 else None
        out_channel, in_channel, kernel_h, kernel_w = conv.weight.size()
        if bn:
            assert bn.bias.size()[0] == out_channel, "conv and bn is not paired"
            # Bias
            bn_b = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(bn.bias)
            bn.bias.data.copy_(bn_b)
            ptr += out_channel
            # Weight
            bn_w = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(bn.weight)
            bn.weight.data.copy_(bn_w)
            ptr += out_channel
            # Running Mean
            bn_rm = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(bn.running_mean)
            bn.running_mean.data.copy_(bn_rm)
            ptr += out_channel
            # Running Var
            bn_rv = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(bn.running_var)
            bn.running_var.data.copy_(bn_rv)
            ptr += out_channel
        else:
            # Load conv. bias
            conv_b = torch.from_numpy(weights[ptr:ptr + out_channel]).view_as(conv.bias)
            conv.bias.data.copy_(conv_b)
            ptr += out_channel
        # Load conv. weights
        num_w = conv.weight.numel()
        conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv.weight)
        conv.weight.data.copy_(conv_w)
        ptr += num_w
    print("parsed:", ptr)
    print("succeed.")    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_yolov3_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = DefaultTrainer.build_model(cfg)
    modules = model.get_conv_bn_modules()
    for m in modules:
        print(m.weight.size())
    load_darknet_weights(args.initial_weights, modules)
    save_path = os.path.join(args.output_dir, "yolov3.pth")
    torch.save(model.state_dict(), save_path)
    print("model save to", save_path)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--initial_weights", metavar="FILE", help="path to initial weights file")
    parser.add_argument("--output_dir", help="dir to save weights file")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
