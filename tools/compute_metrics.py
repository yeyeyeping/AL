import numpy as np
from scipy import ndimage
import sys
import os
import os.path as osp
import json
from skimage.io import imread


def get_edge_points(img):
    """
    Get edge points of a binary segmentation result.

    :param img: (numpy.array) a 2D or 3D array of binary segmentation.
    :return: an edge map.
    """
    dim = len(img.shape)
    if (dim == 2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)
    ero = ndimage.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_assd(s, g, spacing=None):
    """
    Get the Average Symetric Surface Distance (ASSD) between a binary segmentation
    and the ground truth.

    :param s: (numpy.array) a 2D or 3D binary image for segmentation.
    :param g: (numpy.array) a 2D or 2D binary image for ground truth.
    :param spacing: (list) A list for image spacing, length should be 2 or 3.

    :return: The ASSD value.
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim == len(g.shape))
    if (spacing == None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim == len(spacing))
    s_dis = ndimage.distance_transform_edt(1 - s_edge, sampling=spacing)
    g_dis = ndimage.distance_transform_edt(1 - g_edge, sampling=spacing)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


def binary_dice(s, g):
    """
    Calculate the Dice score of two N-d volumes for binary segmentation.

    :param s: The segmentation volume of numpy array.
    :param g: the ground truth volume of numpy array.
    :param resize: (optional, bool)
        If s and g have different shapes, resize s to match g.
        Default is `True`.

    :return: The Dice value.
    """
    assert (len(s.shape) == len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0 * s0 + 1e-5) / (s1 + s2 + 1e-5)
    return dice


def metrics(pred, gt, class_num):
    class_dice = []
    class_assd = []
    for i in range(class_num):
        p, g = (pred == i), (gt == i)
        class_dice.append(binary_dice(p, g))
        class_assd.append(binary_assd(p, g))
    return class_dice, class_assd


assert len(sys.argv) - 1 == 3, "pred folder, mask_folder,class_num"

pred_folder, gt_folder, class_num = sys.argv[1], sys.argv[2], int(sys.argv[3])

names = [f for f in os.listdir(pred_folder) if f.endswith(".png")]
if not len(names):
    names = [f for f in os.listdir(pred_folder) if f.endswith(".npy")]
val_dice, val_assd = [], []
val_json = {}
case = []
for name in names:
    pred_path, gt_path = osp.join(pred_folder, name), osp.join(gt_folder, name)
    if pred_path.endswith(("jpg", "png")):
        pred, gt = imread(pred_path), imread(gt_path)
    elif pred_path.endswith("npy"):
        pred, gt = np.load(pred_path), np.load(gt_path)
    else:
        raise NotImplemented
    class_dice, class_assd = metrics(pred, gt, class_num=class_num)
    val_assd.append(class_assd)
    val_dice.append(class_dice)
    case.append({
        "filename": name,
        "class_dice": class_dice,
        "mean_dice": round(sum(class_dice[1:]) / (len(class_dice) - 1), 4),
        "class_asdd": class_assd,
        "mean_assd": round(sum(class_assd[1:]) / (len(class_dice) - 1), 4)
    })

d, a = np.array(val_dice), np.array(val_assd)
m_assd, m_dice = a[:, 1:].mean(1), d[:, 1:].mean(1)

val_json["metrics"] = {
    "assd_class": {
        str(i): f"{np.round(np.mean(a[:, i]), 4)}~{np.round(np.std(a[:, i]), 4)}"
        for i in range(class_num)
    },
    "case_assd": {
        "mean": np.round(np.mean(m_assd), 4),
        "std": np.round(np.std(m_assd), 4),
    },
    "dice_class": {
        str(i): f"{np.round(np.mean(d[:, i]), 4)}~{np.round(np.std(d[:, i]), 4)}"
        for i in range(class_num)
    },
    "case_dice": {
        "mean": np.round(np.mean(m_dice), 4),
        "std": np.round(np.std(m_dice), 4)
    },

}
sort_case = sorted(case, key=lambda x: x["class_dice"])

val_json["case"] = sort_case
outpath = f"{pred_folder}/../evaluation.json"
if osp.exists(outpath):
    outpath = f"{pred_folder}/../evaluation_latest.json"
with open(outpath, "w") as fp:
    json.dump(val_json, fp, indent=4)
