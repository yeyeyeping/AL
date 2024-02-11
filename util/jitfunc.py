from scipy.ndimage import binary_erosion, generate_binary_structure
from util import get_largest_k_components, label_smooth
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.functional import pairwise_distance


@torch.jit.script
def margin_confidence(
    model_output: torch.Tensor, weight=torch.Tensor([1])) -> torch.Tensor:
    if weight.device != model_output.device:
        weight = weight.to(model_output.device)
    model_output, _ = torch.sort(model_output, dim=1, descending=True)
    weight_socre = torch.abs(model_output[:, 0] - model_output[:, 1]) * weight
    return weight_socre.mean(dim=(-1, -2))


@torch.jit.script
def least_confidence(
    model_output: torch.Tensor, weight=torch.Tensor([1])) -> torch.Tensor:
    if weight.device != model_output.device:
        weight = weight.to(model_output.device)
    output_max = torch.max(model_output, dim=1)[0] * weight
    return output_max.mean(dim=(-2, -1))


@torch.jit.script
def max_entropy(
    model_output: torch.Tensor, weight=torch.Tensor([1])) -> torch.Tensor:
    if weight.device != model_output.device:
        weight = weight.to(model_output.device)
    weight_score = -model_output * torch.log(model_output + 1e-7)
    return torch.mean(weight_score.mean(1) * weight, dim=(-2, -1))


def hisgram_entropy(model_output: torch.Tensor, weight=torch.Tensor([1])):
    if weight.device != model_output.device:
        weight = weight.to(model_output.device)
    score = []
    for output in weight:
        frequency, _ = torch.histogram(output, bins=10)
        probs = frequency / frequency.sum()
        entropy = torch.nansum(-probs * torch.log(probs + 1e-7))
        score.append(entropy)
    return torch.tensor(score)


# @torch.jit.script
def JSD(data: torch.Tensor) -> torch.Tensor:
    # data:round x batch x class x height x width
    mean = data.mean(0)
    # mean entropy per pixel
    mean_entropy = -torch.mean(mean * torch.log(mean + 1e-7), dim=[-3, -2, -1])
    sample_entropy = - \
        torch.mean(torch.mean(data * torch.log(data + 1e-7),
                   dim=[-3, -2, -1]), dim=0)
    return mean_entropy - sample_entropy


@torch.jit.script
def snd(pred: torch.Tensor):
    c, b = pred.shape[1], pred.shape[0]
    pred = pred.permute(0, 2, 3, 1).reshape(b, -1, c)
    class_sim = pred[:, :, None, :] * pred[:, :, :, None]
    en = -torch.sum(class_sim * torch.log(class_sim + 1e-5), [1, 2, 3])
    return en


def var(model_output: torch.Tensor):
    model_output = model_output.softmax(2)
    avg_pred = torch.mean(model_output, dim=0) * 0.99 + 0.005
    consistency = torch.zeros(len(model_output[0]), device=model_output.device)
    for aux in model_output:
        aux = aux * 0.99 + 0.005
        var = torch.sum(nn.functional.kl_div(
            aux.log(), avg_pred, reduction="none"), dim=1, keepdim=True)
        exp_var = torch.exp(-var)
        square_e = torch.square(avg_pred - aux)
        c = torch.mean(square_e * exp_var, dim=[-1, -2, -3]) / \
            (torch.mean(exp_var, dim=[-1, -2, -3]) +
                1e-8)
        consistency += c

    return consistency


def circus(mask):
    kernel = generate_binary_structure(2, 1)
    return np.sum(mask - binary_erosion(mask, kernel))


def circu_area_ratio(mask):
    mask = mask.cpu().numpy()
    labels = np.unique(mask)
    avg_cirarea = []
    for l in labels:
        if l == 0:
            continue
        c = (mask == l).astype(np.uint8)
        area = np.sum(c)
        if area == 0:
            continue

        componet = label_smooth(c[None])[0]
        if np.sum(componet) == 0:
            continue
        cir = circus(componet)
        avg_cirarea.append(cir / area)
        # 越大不确定性越高
    return np.random.uniform(
        0.04, 0.06) if len(avg_cirarea) == 0 else np.mean(avg_cirarea)


def car(batch_mask):
    a = []
    for m in batch_mask:
        a.append(circu_area_ratio(m))
    return torch.as_tensor(a)


def class_var_score(pred, image):
    sample_score = []
    for p, i in zip(pred, image):
        label = p > 0.5

        class_center = []
        class_var = []
        for l in label:
            class_pix = i[:, l]
            class_center.append(class_pix.mean(1))

            var = torch.std(class_pix, dim=1).mean()
            class_var.append(var)

        # 类间散度：中心点之间的距离
        class_center = torch.stack(class_center)
        center = class_center.mean(0)
        between_class = pairwise_distance(class_center, center[None]).sum()
        # 类内方差
        inner_var = torch.mean(torch.as_tensor(class_var))
        inner_var = inner_var * 0.999 + 1e-5
        score = between_class / inner_var
        sample_score.append(score)
    return torch.as_tensor(sample_score)

def self_cosine_sim(f):
    norm_f = F.normalize(f, dim=1)
    return torch.matmul(norm_f, norm_f.T).pow(2).mean()


def self_cosine_sim(f):
    norm_f = F.normalize(f, dim=1)
    return torch.matmul(norm_f, norm_f.T).pow(2).mean()


def inner_class_var_outer_class_div_feature(model_output):
    prediction, _, feature = model_output

    prediction = torch.stack(prediction).mean(0)
    feature = feature[0]
    prediction = F.adaptive_avg_pool2d(
        prediction, output_size=feature.shape[2:]).softmax(1)
    numclass = prediction.shape[1]
    mask = prediction > 0.5
    score = []
    for m, f in zip(mask, feature):
        class_center = []
        inner_class_var = []
        for c in range(numclass):
            label = m[c]
            h, w = torch.where(label)
            center = f[:, h, w].mean(1)
            class_center.append(center)
            inner_class_var.append(self_cosine_sim(f[:, h, w]).item())
        inner_class = torch.mean(torch.as_tensor(inner_class_var))
        outer_class = self_cosine_sim(torch.stack(class_center))
        outer_class = outer_class*0.999+1e-5
        score.append(inner_class/outer_class)
    return torch.as_tensor(score)


if __name__ == '__main__':
    import numpy as np

    a = torch.randn(size=(1, 5, 16, 2, 16, 16))
    print(var(a))
