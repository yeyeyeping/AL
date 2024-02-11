from model.MGUnet import MGUNet
from skimage.io import imread
from os.path import join
from util import read_yml
import torch
import sys
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import zoom
import numpy as np
from albumentations import functional
import matplotlib.pyplot as plt
import json

assert len(sys.argv) - 1 == 3, "result folder, test image folder,mask folder"

COLOR_TABLE = [(0, 0, 0), (222, 148, 80), (147, 71, 238), (187, 19, 208), (98, 43, 249), (166, 58, 136), (202, 45, 114),
               (103, 209, 30), (235, 57, 90), (18, 14, 75), (209,
                                                             156, 101), (230, 13, 166), (200, 150, 134),
               (242, 6, 88), (250, 186, 207), (144, 173,
                                               28), (232, 28, 225), (27, 245, 217), (82, 143, 39),
               (168, 222, 137), (121, 126, 101), (36, 222,
                                                  26), (96, 52, 68), (14, 120, 179), (132, 237, 87),
               (119, 189, 121), (11, 51, 121), (6, 62, 102), (99,
                                                              164, 25), (203, 156, 88), (246, 212, 161),
               (216, 96, 246), (227, 98, 136), (243, 246,
                                                208), (171, 20, 38), (45, 57, 144), (35, 71, 130),
               (162, 204, 152), (80, 192, 81), (80, 68, 237), (54,
                                                               200, 25), (89, 126, 121), (229, 73, 134),
               (156, 249, 101), (253, 73, 35), (13, 54, 156), (187,
                                                               142, 9), (55, 33, 114), (160, 135, 174),
               (182, 187, 236), (64, 80, 115), (58, 218,
                                                247), (20, 107, 222), (7, 83, 48), (217, 193, 130),
               (233, 102, 178), (125, 226, 119), (39, 54,
                                                  158), (106, 193, 45), (174, 98, 216), (21, 38, 98),
               (135, 147, 55), (211, 157, 122), (51, 128, 146), (181,
                                                                 163, 89), (145, 87, 153), (239, 130, 152),
               (138, 71, 107), (170, 186, 210), (19, 196, 2), (248,
                                                               167, 29), (209, 182, 193), (177, 226, 23),
               (214, 223, 197), (74, 187, 51), (60, 12, 39), (94,
                                                              234, 136), (188, 154, 128), (155, 30, 210),
               (225, 40, 55), (36, 3, 222), (43, 31, 148), (109,
                                                            206, 121), (74, 209, 56), (184, 20, 170),
               (83, 152, 69), (43, 97, 155), (166, 13, 176), (97,
                                                              58, 36), (35, 109, 26), (21, 27, 163), (30, 181, 88),
               (27, 54, 18), (54, 96, 202), (59, 170, 176), (124,
                                                             32, 10), (60, 252, 230), (240, 75, 163),
               (110, 184, 56), (126, 204, 158), (210, 149,
                                                 89), (83, 33, 33), (91, 156, 207), (159, 98, 144),
               (147, 67, 249), (65, 112, 172), (8, 94,
                                                186), (106, 252, 95), (56, 235, 12), (35, 16, 64),
               (198, 238, 160), (5, 204, 214), (30, 98, 242), (36,
                                                               253, 61), (73, 144, 213), (53, 130, 252),
               (243, 75, 41), (18, 150, 252), (40, 117, 236)]
mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)

result_folder, img_folder, mask_folder = Path(
    sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
input_size = 192

torch.set_grad_enabled(False)


def model_from_cfg(cfg_path):
    cfg = read_yml(cfg_path)
    model = MGUNet(cfg["Network"]).to(device)
    model.eval()
    return model


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
    for i in range(class_num):
        p, g = (pred == i), (gt == i)
        class_dice.append(binary_dice(p, g))
    return class_dice


imgs = list(Path(img_folder).glob('*.npy'))
result = {}
for method in result_folder.iterdir():
    if not method.is_dir():
        continue
    model = model_from_cfg(str(method / "config.yml"))
    result[method.name] = []
    for ckpoint in sorted(list((method / "checkpoint").glob("c[0,1,2,3,4,5,6,7]_best*"))):
        dice_list = []
        model.load_state_dict(torch.load(str(ckpoint))["model_state_dict"])
        for img_path in imgs:
            mask_path = join(mask_folder, img_path.name)
            img_npy = np.load(str(img_path))
            img_npy = functional.normalize(img_npy, mean, std,max_pixel_value=1)
            zoomed_img = torch.from_numpy(
                img_npy).cuda().permute(2, 0, 1).unsqueeze(0)
            output, _, _ = model(zoomed_img)
            output = torch.stack(output).mean(0)
            batch_pred_mask = output.argmax(axis=1)[0]

            gt_npy = np.load(mask_path)
            dice = np.mean(metrics(batch_pred_mask.cpu().numpy(),
                           gt_npy, class_num=np.max(gt_npy)+1)[1:])
            dice_list.append(dice)
        result[method.name].append(np.mean(dice_list))
    print(method.name, result[method.name])
    with open(str(result_folder / "plot.json"), "w+") as fp:
        json.dump(result, fp, indent=4, ensure_ascii=False)


NUM_COLORS = len(result.keys())
fig, ax = plt.subplots(figsize=(12, 9))
x = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
for i, (k, v) in enumerate(result.items()):
    ax.plot(x, v, label=k, linewidth=3.0, color=[
            i/255 for i in COLOR_TABLE[i*5+9]])
    ax.set_xlabel("Proportion of Labeled Images")
    ax.set_ylabel("DSC")
ax.legend()
plt.savefig(str(result_folder / "plot.png"))
plt.show()
