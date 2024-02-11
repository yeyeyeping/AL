# import numpy as np
# import torch
# import os
# from util import read_yml
# import sys
# from model.MGUnet import MGUNet
# from pathlib import Path
# import SimpleITK as sitk
# from scipy.ndimage import zoom

# assert len(sys.argv) - 1 == 4, "cfg_path, img_folder, ckpath, out_dir"
# cfg_path, img_folder, ckpath, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# assert os.path.exists(img_folder) and os.path.exists(ckpath)

# input_size = 192
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# os.makedirs(out_dir, exist_ok=True)

# cfg = read_yml(cfg_path)
# model = MGUNet(cfg["Network"]).to(device)

# model.load_state_dict(torch.load(ckpath, map_location=device)["model_state_dict"])
# model.eval()
# with torch.no_grad():
#     gt = Path(img_folder).glob("*_gt.nii.gz")
#     for g in gt:
#         img_path = str(g)[:-10] + ".nii.gz"
#         img = sitk.ReadImage(img_path)
#         img_npy = sitk.GetArrayFromImage(img)[:, None]

#         *_, h, w = img_npy.shape
#         zoomed_img = zoom(img_npy, (1, 1, input_size / h, input_size / w), order=1,
#                           mode='nearest')
#         zoomed_img = torch.from_numpy(zoomed_img).cuda()
#         output, _, _ = model(zoomed_img)
#         output = torch.stack(output).mean(0)
#         pred_volume = zoom(output.cpu().numpy(), (1, 1, h / input_size, w / input_size), order=1,
#                            mode='nearest')
#         batch_pred_mask = pred_volume.argmax(axis=1)
#         np.save(os.path.join(out_dir, str(g.name)), batch_pred_mask)


import os
import sys
from pathlib import Path
from albumentations import functional
import numpy as np
import torch

from scipy.ndimage import zoom
from scipy import ndimage
from model.MGUnet import MGUNet
from util import read_yml
from util.reader import reader
from skimage.io import imsave
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
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        gt = gt.cpu().numpy()
    class_dice = []
    class_assd = []
    for i in range(class_num):
        p, g = (pred == i), (gt == i)
        class_dice.append(binary_dice(p, g))
        class_assd.append(binary_assd(p, g))
    return class_dice, class_assd


mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)


assert len(sys.argv) - \
    1 == 6, "cfg_path, img_folder, ckpath, out_dir,input_size,class_num"
cfg_path, img_folder, ckpath, out_dir, input_size, class_num = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(
    sys.argv[5]), int(sys.argv[6])

assert os.path.exists(img_folder) and os.path.exists(ckpath)

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

os.makedirs(out_dir, exist_ok=True)

cfg = read_yml(cfg_path)
model = MGUNet(cfg["Network"]).to(device)

model.load_state_dict(torch.load(
    ckpath, map_location=device)["model_state_dict"])
model.eval()


val_dice, val_assd = [], []
val_json = {}
case = []


with torch.no_grad():
    for g, gt in zip((Path(img_folder)/"images").iterdir(), (Path(img_folder)/"mask").iterdir()):
        print(f"predicting {str(g)}")
        img_npy = reader(g)().read_image(g)
        mask_npy = reader(g)().read_image(gt)

        img_npy = functional.normalize(img_npy, mean, std)
        img_npy = np.ascontiguousarray(img_npy.transpose(2, 0, 1))[None]

        img = torch.from_numpy(img_npy).to(device, torch.float32)
        output, _, _ = model(img)
        output = torch.stack(output).mean(0)
        batch_pred_mask = output.argmax(axis=1)[0]
        imsave(os.path.join(out_dir, str(g.name)[
               :-4]+".png"), batch_pred_mask.cpu().numpy(), check_contrast=False)

        class_dice, class_assd = metrics(
            batch_pred_mask, mask_npy, class_num=class_num)
        val_assd.append(class_assd)
        val_dice.append(class_dice)
        case.append({
            "filename": g.name,
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
outpath = f"{os.path.dirname(ckpath)}/../evaluation.json"
with open(outpath, "w") as fp:
    json.dump(val_json, fp, indent=4)
