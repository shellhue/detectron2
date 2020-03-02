import os
import re
import numpy as np


def get_subfolders_recursively(folder_path):
    """Get all subfolders recursively in folder_path.
    """
    folder_list = []
    for root, dirs, _ in os.walk(folder_path):
        for one_dir in dirs:
            one_dir = os.path.join(root, one_dir)
            folder_list.append(one_dir)
    return folder_list


def _get_direct_files_in_dir(dir_path, formats):
    """Get all direct imgs in dir_path.
    """
    imgs = []
    files = os.listdir(dir_path)
    for file in files:
        f = os.path.splitext(file)[1]
        if f in formats:
            imgs.append(os.path.join(dir_path, file))
    return imgs


def get_all_imgs_in_dir(root_dir):
    """Get all imgs recursively in dir.
    """
    all_imgs = []
    subfolders = get_subfolders_recursively(root_dir)
    subfolders.append(root_dir)
    for folder in subfolders:
        imgs = _get_direct_files_in_dir(
            folder, [".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"])
        all_imgs.extend(imgs)
    return all_imgs


def get_all_xmls_in_dir(root_dir):
    """Get all xmls recursively in dir.
    """
    all_xmls = []
    subfolders = get_subfolders_recursively(root_dir)
    subfolders.append(root_dir)
    for folder in subfolders:
        xmls = _get_direct_files_in_dir(folder, [".xml"])
        all_xmls.extend(xmls)
    return all_xmls


def get_all_weights_file_in_dir(root_dir):
    """Get all weights file recursively in dir.
    """
    all_files = []

    subfolders = get_subfolders_recursively(root_dir)
    subfolders.append(root_dir)
    for folder in subfolders:
        f = _get_direct_files_in_dir(folder, [".h5", ".hdf5"])
        all_files.extend(f)
    return all_files


def get_best_saved_weights_file(root_dir, epoch_reg=r"checkpoint-(\d+).h5"):
    if not os.path.isdir(root_dir):
        return "", 0
    ws = get_all_weights_file_in_dir(root_dir)
    epoch = 0
    weights = ""
    for w in ws:
        res = re.search(epoch_reg, w)
        e = (int)(res.group(1))
        if e > epoch:
            epoch = e
            weights = w
    return weights, epoch


def get_classes(classes_path):
    """Read classes info from classes path.

    Arguments:
        classes_path (str): Classes path.

    Returns: 
        A list: A list of class names with the same order in classes file.
    """
    with open(classes_path, encoding="utf-8") as f:
        lines = f.readlines()
    class_names = []
    for l in lines:
        cls = l.strip()
        if len(cls) > 0:
            class_names.append(cls)

    return class_names


def get_class_id_to_name(class_names, start=0):
    """Get class id to name from list of class names.

    Arguments:
        class_names (list of string): The list of class names in order of their id.
        start (int): The start id of the first class name.

    Returns:
        A dict: Key is class id, value is the class corresponding name.
    """
    id2name = {}
    for i, n in enumerate(class_names):
        id2name[i + start] = n
    return id2name


def get_anchors(anchor_path):
    """Load all anchors form path.

    Arguments:

        anchor_path (str): The anchor file path

    Returns: 
        A 2d numpy array: Shape is [N, 2]. N is the number of anchor.
    """
    with open(anchor_path) as f:
        first_line = f.readline()
    anchors = []

    for a in first_line.split(','):
        anchor = a.strip()
        if len(anchor) > 0:
            anchors.append(anchor)
    anchors = np.array(anchors, dtype='float32').reshape([-1, 2])
    return anchors


if __name__ == "__main__":
    classes = get_classes(
        "/Users/huangzeyu/Desktop/easydet/data/dataset/coco/classes.txt")
    print(len(classes))
    print(classes)
    # anchors = get_anchors(
    #     "/Users/huangzeyu/Desktop/easydet/models/yolov3/yolo_anchors.txt")
    # print(anchors)
