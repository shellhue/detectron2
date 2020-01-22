import os
import re
import numpy as np


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


def check_path_end(path):
    """Make sure path ends with /.
    """
    return path if path.endswith('/') else path + '/'


def get_subfolders_recursively(folder_path):
    """Get all subfolders recursively in folder_path.
    """
    folder_list = []
    for root, dirs, _ in os.walk(folder_path):
        for one_dir in dirs:
            one_dir = os.path.join(root, one_dir)
            folder_list.append(one_dir)
    return folder_list


# if __name__ == "__main__":
#     classes = get_classes(
#         "/Users/huangzeyu/Desktop/easydet/data/dataset/coco/classes.txt")
#     print(len(classes))
#     print(classes)
    # anchors = get_anchors(
    #     "/Users/huangzeyu/Desktop/easydet/models/yolov3/yolo_anchors.txt")
    # print(anchors)
