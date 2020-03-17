# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer

class CustomDetectionCheckpointer(DetectionCheckpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """
    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            super()._load_model(checkpoint)
        else:
            model_state_dict = self.model.state_dict()
            ckpt_model_dict = checkpoint["model"]
            from_backbone = False
            new_ckpt_state_dict = {}
            for k, v in ckpt_model_dict.items():
                components = k.split(".")
                if len(components) > 1:
                    components.insert(1, "bottom_up")
                    nk = ".".join(components)
                    if nk in model_state_dict:
                        from_backbone = True
                        new_ckpt_state_dict[nk] = v.clone()
                if not from_backbone:
                    new_ckpt_state_dict[k] = v.clone()
            # checkpoint["model"] = new_ckpt_state_dict
            # print(checkpoint.keys())
            # assert False
            if True:
                checkpoint.clear()
                checkpoint["model"] = new_ckpt_state_dict
            super()._load_model(checkpoint)
        

