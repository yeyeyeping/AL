import os
import sys
from pathlib import Path
from albumentations import functional
import numpy as np
import torch
from util import read_yml
from reader import reader
from skimage.io import imsave
import numpy as np
import sys
import os
import json
from util import trainer
from metric import metrics
from model.MGUnet import MGUNet
from pymic.util.image_process import get_largest_k_components

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)

tolerance = 10


def label_smooth(seg):
    cls_num = np.unique(seg)
    output = np.zeros_like(seg)
    for c in cls_num:
        if c == 0:
            continue
        seg_c = np.asarray(seg == c, np.uint8)
        seg_c = get_largest_k_components(seg_c)
        output = output + seg_c * c
    return output

assert len(sys.argv) - \
    1 == 6, "cfg_path, img_folder, ckpath, out_dir,class_num,include_back"
cfg_path, img_folder, ckpath, out_dir, class_num, include_back = sys.argv[
    1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), eval(
        sys.argv[6])
assert os.path.exists(img_folder) and os.path.exists(ckpath)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")

os.makedirs(out_dir, exist_ok=True)

cfg = read_yml(cfg_path)
model = MGUNet(cfg["Network"]).to(device)

model.load_state_dict(
    torch.load(ckpath, map_location=device)["model_state_dict"])
model.eval()

val_dice, val_assd = [], []
val_json = {}
case = []

with torch.no_grad():
    for g, gt in zip((Path(img_folder) / "images").iterdir(),
                     (Path(img_folder) / "mask").iterdir()):
        print(f"predicting {str(g)}")
        img_npy = reader(g)().read_image(g)
        mask_npy = reader(g)().read_image(gt)

        img_npy = functional.normalize(img_npy, mean, std, max_pixel_value=1)
        img_npy = np.ascontiguousarray(img_npy.transpose(2, 0, 1))[None]

        img = torch.from_numpy(img_npy).to(device, torch.float32)
        output, _, _ = model(img)
        output = torch.stack(output).mean(0)

        # output = model(img)
        batch_pred_mask = output.argmax(axis=1)[0].cpu().numpy().astype(
            np.uint8)

        imsave(os.path.join(out_dir,
                            str(g.name)[:-4] + ".png"),
               batch_pred_mask,
               check_contrast=False)
        class_dice, class_assd = metrics(batch_pred_mask,
                                         mask_npy,
                                         include_back=include_back,
                                         class_num=class_num,
                                         tolerance=tolerance)
        # class_dice, class_assd = class_dice, class_assd
        val_assd.append(class_assd)
        val_dice.append(class_dice)
        case.append({
            "filename": g.name,
            "class_dice": class_dice,
            "mean_dice": round(sum(class_dice) / (len(class_dice)), 2),
            "class_asdd": class_assd,
            "mean_assd": round(sum(class_assd) / (len(class_dice)), 2)
        })

d, a = np.array(val_dice), np.array(val_assd)
m_assd, m_dice = a.mean(1), d.mean(1)

val_json["metrics"] = {
    "assd_class": {
        str(i):
        f"{np.round(np.mean(a[:, i]), 2)}±{np.round(np.std(a[:, i]), 2)}"
        for i in range(a.shape[1])
    },
    "case_assd":
    f"{np.round(np.mean(m_assd), 2)}±{np.round(np.std(m_assd), 2)}",
    "dice_class": {
        str(i):
        f"{np.round(np.mean(d[:, i]), 2)}±{np.round(np.std(d[:, i]), 2)}"
        for i in range(a.shape[1])
    },
    "case_dice":
    f"{np.round(np.mean(m_dice), 2)}±{np.round(np.std(m_dice), 2)}"
}
sort_case = sorted(case, key=lambda x: x["mean_dice"])

val_json["case"] = sort_case
outpath = f"{out_dir}/../{os.path.basename(out_dir)}.json"
with open(outpath, "w", encoding="utf-8") as fp:
    json.dump(val_json, fp, indent=4, ensure_ascii=False)
