from os.path import join as osjoin
from os.path import exists as osexists
from torch.nn.functional import one_hot
from os.path import basename
from torch.utils.data import Dataset, DataLoader
import os
import ast
from typing import Any
from albumentations import functional
from util import read_yml
from model.MGUnet import MGUNet
from util.reader import reader
import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
import time
from sklearn import manifold
import albumentations as A
from util import SubsetSampler
from pathlib import Path
from util import get_dataloader_ISIC
from dataset.ACDCDataset import ISICDataset
from util import jitfunc as f


COLORS = ['aqua', 'midnightblue', 'red', 'darkgreen',
          'darkred', 'maroon', 'purple', 'indigo', 'darkslategray', 'black']


def dc(pred, label):
    pred, label = pred > 0.5, one_hot(label.squeeze(
        1), pred.shape[1]).permute(0, 3, 1, 2).bool()
    tp = torch.sum(pred == label, dim=[-1, -2])
    fp = torch.sum(pred == torch.logical_not(label), dim=[-1, -2])
    fn = torch.sum(torch.logical_not(pred) == label, dim=[-1, -2])

    nominator = 2 * tp
    denominator = 2 * tp + fp + fn

    dc = (nominator + 1e-5) / (torch.clip(denominator + 1e-5, 1e-8))
    return dc[:, 1:].mean(1)


def build_model(cfg):
    device = cfg["Training"]["device"]
    model = MGUNet(cfg["Network"]).to(device)
    return model


class FeatureExtractor256:
    def to(self, device):
        self.feature.to(device)

    def __init__(self, pool_size=12):
        self.device = "cuda"
        self.feature = self.build_feature_layer(pool_size=pool_size)

    def build_feature_layer(self, pool_size):
        d = 512

        pool = nn.AdaptiveAvgPool2d(
            (pool_size, pool_size)).to(self.device)
        con1x1 = nn.Conv2d(d, d,
                           kernel_size=pool_size, bias=False).to(self.device)
        relu = nn.ReLU(True)
        bn = nn.BatchNorm2d(d).to(self.device)
        feature = nn.Sequential(pool, con1x1, bn, relu)
        feature.eval()
        return feature

    def __call__(self, model_output):
        _, _, features = model_output
        return self.feature(features[0]).flatten(
            1, -1).cpu().numpy()


class DefeultFeatureExtractor:
    def to(self, device):
        self.pool.to(device)

    def __init__(self, pool_size=12):
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((pool_size, pool_size)))

    def __call__(self, model_output):
        _, _, features = model_output
        return self.pool(features[0]).view((features[0].shape[0], -1)).cpu().numpy()


class MultiLevelFeatureExtractor:
    def to(self, device):
        self.pool.to(device)

    def __init__(self, pool_size=12):
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((pool_size, pool_size)))

    def __call__(self, model_output):
        _, _, features = model_output

        return torch.concat([self.pool(i) for i in features], dim=1).view((features[0].shape[0], -1)).cpu().numpy()


class IDataset(Dataset):
    def __init__(self, trainfolder, transform=None) -> None:
        super().__init__()
        self.folder = trainfolder
        self.images = list((Path(trainfolder) / "images").glob("*.npy"))
        self.transforms = transform

    def __getitem__(self, index):
        data = np.load(str(self.images[index]))

        if self.transforms is not None:
            transformed = self.transforms(image=data)
            data = transformed["image"]

        return torch.tensor(np.transpose(data, [2, 0, 1]))

    def __len__(self):
        return len(self.images)


def entropy(model_output, img):
    model_output, _, _ = model_output
    if len(model_output) == 1:
        o = model_output[0]
    elif len(model_output) > 1:
        o = model_output.mean(0)
    else:
        raise NotImplementedError
    return f.max_entropy(o.softmax(dim=1))


def leastconfidence(model_output, img):
    model_output, _, _ = model_output
    if len(model_output) == 1:
        o = model_output[0]
    elif len(model_output) > 1:
        o = model_output.mean(0)
    else:
        raise NotImplementedError
    return f.least_confidence(o.softmax(dim=1))


def marginconfidence(model_output, img):
    model_output, _, _ = model_output
    if len(model_output) == 1:
        o = model_output[0]
    elif len(model_output) > 1:
        o = model_output.mean(0)
    else:
        raise NotImplementedError
    return f.margin_confidence(o.softmax(dim=1))


def var(model_output, img):
    model_output = torch.stack(model_output[0]).softmax(2)
    avg_pred = torch.mean(model_output, dim=0) * 0.99 + 0.005
    consistency = torch.zeros(len(model_output[1]), device=avg_pred.device)
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


def class_var(model_output, img):
    model_output, _, _ = model_output
    if len(model_output) == 1:
        o = model_output[0]
    elif len(model_output) > 1:
        o = torch.stack(model_output).mean(0)
    else:
        raise NotImplementedError
    return f.class_var_score(o, img)


def var(model_output, img):
    model_output = torch.stack(model_output[0]).softmax(2)
    avg_pred = torch.mean(model_output, dim=0) * 0.99 + 0.005
    consistency = torch.zeros(len(model_output[1]), device=model_output.device)
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


class ALVisualization():

    def __init__(self, training_dir):
        self.base_dir = training_dir
        query_file = osjoin(self.base_dir, 'query_state')
        self.init_labeled, self.labeled, self.init_unlabeled, self.unlabeled = self._get_query_record(
            query_file)
        ckdir = osjoin(self.base_dir, 'checkpoint')
        self.ckpoints = sorted([osjoin(ckdir, i) for i in os.listdir(
            ckdir) if "best" in i], key=lambda x: int(basename(x).split('_')[0][1:]))
        self.cycle_num = len(self.ckpoints)
        self.cfg = read_yml(osjoin(self.base_dir, "config.yml"))

    def _dif(self, a):
        r = [a[0], ]
        for i in range(1, len(a)):
            if len(a[i]) > len(a[i-1]):
                r.append(a[i] - a[i-1])
            else:
                r.append(a[i-1] - a[i])
        return r

    def _get_query_record(self, filepath):
        if not osexists(filepath):
            raise FileNotFoundError

        with open(filepath) as fp:
            indices = fp.readlines()
        lab, unlab = indices[::2], indices[1::2]

        lab, unlab = [set(ast.literal_eval(l.strip())) for l in lab], \
            [set(ast.literal_eval(l.strip())) for l in unlab]
        # print(unlab)
        lab, unlab = self._dif(lab), self._dif(unlab)

        return lab[0], lab[1:], unlab[0], unlab[1:]

    @torch.no_grad()
    def embedding(self, model, loader, feature_extractor):
        model.eval()
        device = next(iter(model.parameters())).device

        features = []
        for img in loader:
            img = img.to(device, torch.float32)
            model_output = model(img)
            features += list(feature_extractor(model_output))

        return features

    def imgdir2loader(self, img_dir):
        class SimpleDataset(Dataset):
            def __init__(self, img_dir) -> None:
                self.imgs = [osjoin(img_dir, i) for i in os.listdir(img_dir)]
                self.reader = reader(self.imgs[0])()

            def __len__(self) -> int:
                return len(self.imgs)

            def __getitem__(self, index) -> Any:
                i = self.imgs[index]
                img_npy = self.reader.read_image(i)
                img_npy = functional.normalize(
                    img_npy, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=1)
                img_npy = np.ascontiguousarray(
                    img_npy.transpose(2, 0, 1))
                return torch.tensor(img_npy)

        loader = DataLoader(SimpleDataset(img_dir=img_dir),
                            batch_size=8, pin_memory=True, num_workers=4)
        return loader

    def tsne_vis_features(self, cycle, img_dirs=None, loaders=None, ax=None, colors="blue", labels="initial_set", feature_extractor=None, out_path=None, save=False):

        if type(img_dirs) == str:
            img_dirs = [img_dirs, ]
        if type(loaders) == DataLoader:
            loaders = [loaders, ]

        if type(colors) == str:
            colors = [colors, ]

        if type(labels) == str:
            labels = [labels, ]

        if img_dirs is not None and loaders is None:
            loaders = []
            for imgdir in img_dirs:
                loaders.append(self.imgdir2loader(imgdir))

        if img_dirs is None and loaders is None:
            raise f"img_dirs =None, loaders None"

        if len(colors) != len(loaders):
            raise f"len(img_dirs) == len(loaders)"

        if feature_extractor == None:
            feature_extractor = DefeultFeatureExtractor()

        model = build_model(self.cfg)

        if cycle > self.cycle_num:
            raise ValueError

        model.load_state_dict(
            torch.load(self.ckpoints[cycle], map_location=self.cfg["Training"]["device"])["model_state_dict"])

        embeddings = []
        for loader in loaders:
            embeddings.append(self.embedding(model, loader, feature_extractor))

        tsne = manifold.TSNE(n_components=2, init='pca',
                             random_state=319).fit_transform(np.concatenate(embeddings))
        splits = np.cumsum([len(e) for e in embeddings])[:-1]

        tsne_splits = np.split(tsne, splits)

        if ax is None:
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"cycle {cycle}")
        for s, c in zip(tsne_splits, colors):
            ax.scatter(s[:, 0], s[:, 1], s=10, c=c)

        ax.legend(labels)
        if save:
            if out_path is None:
                out_path = osjoin(self.base_dir, f"{time.time()}.png")
            ax.figure.savefig(out_path)

        return ax

    def build_loader(self, idxs, with_label=False):
        sampler = SubsetSampler(list(idxs))
        data_dir = self.cfg["Dataset"]["data_dir"]
        if not with_label:
            dataset = IDataset(trainfolder=osjoin(data_dir, "train"))
        elif with_label:
            dataset = ISICDataset(trainfolder=osjoin(data_dir, "train"))
        return DataLoader(dataset, batch_size=16,
                          sampler=sampler, persistent_workers=True, prefetch_factor=4,
                          num_workers=4)

    def vis_feature(self, out_fig_path,  feature_extractor=None):
        loaders = [self.build_loader(
            self.init_unlabeled), self.build_loader(self.init_labeled)]
        colors = [COLORS[0], COLORS[1]]
        labels = ["initial_unlab", "initial_lab"]
        fig = plt.figure(figsize=(16, 9))
        w = int(np.ceil(np.sqrt(self.cycle_num)))
        for i, idxs in enumerate(self.unlabeled):
            ax = fig.add_subplot(w, w, i+1)
            l = self.build_loader(idxs)
            loaders.append(l)
            colors.append(COLORS[i+2])
            labels.append(f"query {i+1}")
            self.tsne_vis_features(
                i, ax=ax, loaders=loaders, colors=colors, labels=labels, feature_extractor=feature_extractor)
        fig.savefig(out_fig_path)

    @torch.no_grad()
    def vis_one_consistency(self, ax, cycle, idxs, score_func, color="blue"):
        if cycle > self.cycle_num or cycle < 0:
            raise ValueError

        device = self.cfg["Training"]["device"]

        model = build_model(self.cfg)

        model.load_state_dict(
            torch.load(self.ckpoints[cycle], map_location=device)["model_state_dict"])
        print(f"loading {self.ckpoints[cycle]}")
        loader = self.build_loader(idxs=idxs, with_label=True)

        model.eval()
        score_dc = []
        for (img, label) in loader:
            img, label = img.to(device), label.to(device)
            model_output = model(img)
            # 计算分数
            scores = score_func(model_output, img)
            # 计算和标签的dice
            model_output, _, _ = model_output
            if len(model_output) == 1:
                o = model_output[0]
            elif len(model_output) > 1:
                o = torch.stack(model_output).mean(0)
            else:
                raise NotImplementedError

            dice = dc(o, label)
            score_dc += list(np.stack([scores.cpu().numpy(),
                             dice.cpu().numpy()], axis=1))

        if ax is None:
            plt.figure(figsize=(16, 9))
            ax = plt.subplot()
        score_dc = np.asanyarray(score_dc)
        ax.scatter(score_dc[:, 0], score_dc[:, 1], s=10, c=color)

        return ax

    def vis_consistency(self, score_func, outpath):
        fig = plt.figure(figsize=(16, 9))
        w = int(np.ceil(np.sqrt(self.cycle_num)))

        idxs = [self.init_unlabeled,] + self.unlabeled

        for i in range(self.cycle_num):
            ax = fig.add_subplot(w, w, i + 1)
            ax.set_title(f"cycle {i}")
            legend = ["initial_set", ]
            for j, idx in enumerate(idxs[:i+2]):
                self.vis_one_consistency(
                    ax, i, list(idx), score_func, color=COLORS[j])
                legend.append(f"query {j+1}")
            ax.legend(legend)
        fig.savefig(outpath)


def inner_class_var_outer_class_div(model_output, image):
    return f.inner_class_var_outer_class_div_feature(model_output)


def vis_full(score_func):
    cfg = read_yml(
        "/home/yeep/project/py/deeplearning/AL-ACDC/EXP/ISIC/FULL/config.yml")
    device = cfg["Training"]["device"]
    model = build_model(cfg)
    model.load_state_dict(
        torch.load("/home/yeep/project/py/deeplearning/AL-ACDC/EXP/ISIC/FULL/checkpoint/c0_best0.8970.pt", map_location=device)["model_state_dict"])

    model.eval()
    data_dir = cfg["Dataset"]["data_dir"]
    dataset = ISICDataset(trainfolder=osjoin(data_dir, "train"))
    loader = DataLoader(dataset, batch_size=16,
                        persistent_workers=True, prefetch_factor=4, num_workers=4)
    score_dc = []
    for (img, label) in loader:
        img, label = img.to(device), label.to(device)
        model_output = model(img)
        # 计算分数
        scores = score_func(model_output, img)
        # 计算和标签的dice
        model_output, _, _ = model_output
        if len(model_output) == 1:
            o = model_output[0]
        elif len(model_output) > 1:
            o = torch.stack(model_output).mean(0)
        else:
            raise NotImplementedError

        dice = dc(o, label)
        score_dc += list(np.stack([scores.cpu().numpy(),
                                   dice.cpu().numpy()], axis=1))

    plt.figure(figsize=(16, 9))
    ax = plt.subplot()
    score_dc = np.asanyarray(score_dc)
    ax.scatter(score_dc[:, 0], score_dc[:, 1], s=10, c=COLORS[0])
    return ax


if __name__ == '__main__':
    import sys
    train_dir = sys.argv[1]
    type_ = sys.argv[2]
    vis = ALVisualization(train_dir)
    if type_ == "feature":
        vis.vis_feature(f"{train_dir}/feature.png")
    else:
        score_func = globals()[type_]
        vis.vis_consistency(score_func, f"{train_dir}/{type_}")

    # vis.vis_feature(
    #     "/home/yeep/project/py/deeplearning/AL-ACDC/GMClusterQuery.png", feature_extractor=FeatureExtractor256())
    # vis.vis_consistency(
    #     class_var, "/home/yeep/project/py/deeplearning/AL-ACDC/ClassVarQuery.png")
    # vis.vis_feature("/home/yeep/project/py/deeplearning/AL-ACDC/2.png")
    # fig = plt.figure()

    # ax1 = fig.add_subplot(2, 2, 1)
    # extractor = MultiLevelFeatureExtractor()
    # vis.tsne_vis_features(
    #     0, img_dirs="/home/yeep/project/py/deeplearning/AL-ACDC/data/ISIC/test/images", feature_extractor=extractor, ax=ax1)

    # ax2 = fig.add_subplot(2, 2, 2)
    # vis.tsne_vis_features(
    #     1, img_dirs="/home/yeep/project/py/deeplearning/AL-ACDC/data/ISIC/test/images", feature_extractor=extractor, ax=ax2)

    # ax3 = fig.add_subplot(2, 2, 3)

    # vis.tsne_vis_features(
    #     2, img_dirs="/home/yeep/project/py/deeplearning/AL-ACDC/data/ISIC/test/images", feature_extractor=extractor, ax=ax3)

    # ax4 = fig.add_subplot(2, 2, 4)
    # vis.tsne_vis_features(
    #     3, img_dirs="/home/yeep/project/py/deeplearning/AL-ACDC/data/ISIC/test/images", feature_extractor=extractor, ax=ax4)

    # fig.savefig("2.png")
