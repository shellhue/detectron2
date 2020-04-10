import copy
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from .fetcher import get_classes

def _get_imagenet_metadata(classes_file):
    classes = get_classes(classes_file)
    class_id_to_name = {}
    # class_id_to_short_name = {}
    class_name = []
    class_id = []
    for i, name in enumerate(classes):
        if len(name.split(":")) > 1:
            short_name, human_readale_name = name.split(":")
        else:
            human_readale_name = name
        readable_name = human_readale_name.split(",")[0]
        class_id_to_name[i] = readable_name
        # class_id_to_short_name[i] = short_name
        class_name.append(readable_name)
        class_id.append(i)
    return {
        "class_ids": class_id,
        "thing_classes": class_name,
        "class_id_to_name": class_id_to_name,
        # "class_id_to_short_name": class_id_to_short_name
    }


def load_imagenet_file_info(info_file):
    with open(info_file, encoding="utf-8") as f:
        lines = f.readlines()
    info = []
    for l in lines:
        image_info = l.strip()
        if len(image_info) > 0:
            image_path, class_id = image_info.split(":")
            class_id = int(class_id)
            info.append({
                "file_name": image_path,
                "id": class_id
            })
    return info


def register_imagenet_dataset(name, classes_file, image_info_file, root):
    classes_file = os.path.join(root, classes_file)
    image_info_file = os.path.join(root, image_info_file)
    metadata = _get_imagenet_metadata(classes_file)
    DatasetCatalog.register(
        name, lambda: load_imagenet_file_info(image_info_file))
    MetadataCatalog.get(name).set(
        classes_file=classes_file, image_info_file=image_info_file, evaluator_type="classification", **metadata
    )


_ALL_IMAGENET_SPLITS = {
    "imagenet_train": ("imagenet/train.txt", "imagenet/classes.txt"),
    "imagenet_val": ("imagenet/val_remote.txt", "imagenet/classes.txt")
}

_ALL_SMOKE_PHONE_SPLITS = {
    "smoke_call_train": ("smoke_phone/train.txt", "smoke_phone/classes.txt"),
    "smoke_call_val": ("smoke_phone/val.txt", "smoke_phone/classes.txt")
}

def register_all_imagenet(root="datasets"):
    if not root.startswith("/"):
        file_path = __file__
        components = file_path.split('/')[:-4]
        components.append(root)
        root = "/".join(components)
    for name, (img_info_file, class_file) in _ALL_IMAGENET_SPLITS.items():
        register_imagenet_dataset(name, class_file, img_info_file, root)

def register_all_smoke_phone(root="datasets"):
    if not root.startswith("/"):
        file_path = __file__
        components = file_path.split('/')[:-4]
        components.append(root)
        root = "/".join(components)
    for name, (img_info_file, class_file) in _ALL_SMOKE_PHONE_SPLITS.items():
        register_imagenet_dataset(name, class_file, img_info_file, root)

register_all_imagenet()
register_all_smoke_phone()
