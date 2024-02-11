# from util.vis import build_model, read_yml, DefeultFeatureExtractor
# import torch
# from os.path import join as osjoin
# from torch.utils.data import DataLoader
# from dataset.ACDCDataset import ISICDataset
# from sklearn.mixture import GaussianMixture
# import numpy as np
# from sklearn.covariance import EmpiricalCovariance
# from torch import nn


# class DefeultFeatureExtractor:
#     def to(self, device):
#         self.pool.eval()
#         self.bn.eval()
#         self.con1x1.eval()

#         self.pool.to(device)
#         self.bn.to(device)
#         self.con1x1.to(device)
#         return self

#     def __init__(self, pool_size=12):
#         self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
#         self.bn = nn.BatchNorm2d(512)
#         self.con1x1 = nn.Conv2d(512, 256,
#                                 kernel_size=pool_size, bias=False)

#     def __call__(self, model_output):
#         _, _, features = model_output
#         ret = self.con1x1(self.bn(self.pool(features[0]))).flatten(
#             1, -1).cpu().numpy()
#         return ret


# class MDistance:
#     def __init__(self) -> None:
#         self.ec = EmpiricalCovariance(assume_centered=True)

#     def fit(self, X):
#         self.ec.fit(X)

#     def distance(self, x1, x2):
#         d = x1 - x2
#         return d@self.ec.precision_@d.T


# ckpoint = "/home/yeep/project/py/deeplearning/AL-ACDC/EXP/ISIC/MGUnet_avgalign/checkpoint/c3_best0.8860.pt"
# cfg_path = "/home/yeep/project/py/deeplearning/AL-ACDC/EXP/ISIC/MGUnet_avgalign/config.yml"
# cfg = read_yml(cfg_path)

# model = build_model(cfg)
# model.load_state_dict(torch.load(
#     ckpoint, map_location=cfg["Training"]["device"])["model_state_dict"])
# model.eval()
# torch.set_grad_enabled(False)
# data_dir = cfg["Dataset"]["data_dir"]
# dataset = ISICDataset(trainfolder=osjoin(data_dir, "train"))
# loader = DataLoader(dataset, batch_size=16,
#                     persistent_workers=True, prefetch_factor=4,
#                     num_workers=4)
# embeddings = []
# e = DefeultFeatureExtractor(12).to(cfg["Training"]["device"])
# for img, _ in loader:
#     embeddings.append(e(model(img.cuda())))
# features = np.concatenate(embeddings)
# calc = MDistance()
# calc.fit(features)

# print(calc.ec.precision_.shape)
# print(calc.ec.precision_)


def p():
    print(1)


def p():
    print(2)


p()
