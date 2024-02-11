from pymic.util.evaluation_seg import binary_dice, binary_iou, binary_assd
import torch
from torch.nn import functional as F


def get_metric(pred, gt):
    assert pred.ndim == 3 and gt.ndim == 3
    return binary_dice(pred, gt), binary_iou(pred, gt), binary_assd(pred, gt)


def get_multi_class_metric(pred, gt, class_num, include_backgroud=False):
    dice_list, assd_list, iou_list = [], [], []
    i = 1 if not include_backgroud else 0
    for label in range(i, class_num):
        p, g = pred == label, gt == label
        dice_list.append(round(binary_dice(p, g), 3))
        assd_list.append(round(binary_assd(p, g), 3))
        iou_list.append(round(binary_iou(p, g), 3))

    return dice_list, iou_list, assd_list


def get_classwise_dice(predict, soft_y, pix_w=None):
    """
    Get dice scores for each class in predict (after softmax) and soft_y.

    :param predict: (tensor) Prediction of a segmentation network after softmax.
    :param soft_y: (tensor) The one-hot segmentation ground truth.
    :param pix_w: (optional, tensor) The pixel weight map. Default is None.

    :return: Dice score for each class.
    """

    if (pix_w is None):
        y_vol = torch.sum(soft_y, dim=0)
        p_vol = torch.sum(predict, dim=0)
        intersect = torch.sum(soft_y * predict, dim=0)
    else:
        y_vol = torch.sum(soft_y * pix_w, dim=0)
        p_vol = torch.sum(predict * pix_w, dim=0)
        intersect = torch.sum(soft_y * predict * pix_w, dim=0)
    dice_score = (2.0 * intersect + 1e-5) / (y_vol + p_vol + 1e-5)
    return dice_score


def dc(pred, gt, class_num, include_backgroud=False):
    i = 1 if not include_backgroud else 0

    #onehot
    onehot_gt = F.one_hot(gt, class_num).permute(0, 3, 1, 2)
    pred = F.one_hot(pred, class_num).permute(0, 3, 1, 2)

    pred, gt = pred[:, i:], gt[:, i:]
    intersection = (onehot_gt * pred).sum(dim=(-1, -2))
    union = onehot_gt.sum(dim=(-1, -2)) + pred.sum(dim=(-1, -2))

    return ((2 * intersection + 1e-5) / (union + 1e-5)).mean(0)
