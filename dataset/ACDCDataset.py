import torch
import os
import numpy as np
from os.path import exists
import SimpleITK as sitk
from torch.utils.data import Dataset
from pathlib import Path


class ISICDataset(Dataset):

    def __init__(self, trainfolder, transform=None) -> None:
        super().__init__()
        self.folder = trainfolder
        self.images = list((Path(trainfolder) / "images").glob("*.npy"))
        self.transforms = transform

    def __getitem__(self, index):
        mask_path = str(self.images[index].parent.parent / "mask" /
                        self.images[index].name)
        data, mask = np.load(str(self.images[index])), np.load(mask_path)
        if self.transforms is not None:
            transformed = self.transforms(image=data, mask=mask)
            data, mask = transformed["image"], transformed["mask"]

        return torch.tensor(np.transpose(data, [2, 0, 1]), dtype=torch.float32), \
            torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    def __len__(self):
        return len(self.images)


class ACDCDataset2d(Dataset):

    def __init__(self, trainfolder: str, transform=None) -> None:
        assert os.path.exists(trainfolder), trainfolder
        super().__init__()
        self.data_folder = trainfolder
        self.transforms = transform
        assert exists(trainfolder)
        self.data = list(Path(trainfolder).glob("*gt.npy"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        p = self.data[index]
        imgname = p.name.split("_gt.npy")[0] + ".npy"
        imgpath, labelpath = str(p.parent / imgname), str(p)
        data, label = np.load(imgpath), np.load(labelpath)
        data, label = np.asarray(data,
                                 dtype=np.float32), np.asarray(label,
                                                               dtype=np.uint8)
        data = (data - data.mean()) / data.std()
        if self.transforms is not None:
            transformed = self.transforms(image=data, mask=label)
            data, label = transformed["image"], transformed["mask"]
        return torch.tensor(data,
                            dtype=torch.float32).unsqueeze(0), torch.tensor(
                                label, dtype=torch.long).unsqueeze(0)


class ACDCDataset3d(Dataset):

    def __init__(self, folder) -> None:
        super().__init__()
        assert exists(folder), folder
        self.nii_data = list(Path(folder).glob("*_gt.nii.gz"))

    def __len__(self):
        return len(self.nii_data)

    def __getitem__(self, index):
        p = self.nii_data[index]
        imgname = f"{p.name.split('_gt.nii.gz')[0]}.nii.gz"
        imgpath, labpath = str(p.parent / imgname), str(p)

        img_obj, mask_obj = sitk.GetArrayFromImage(
            sitk.ReadImage(imgpath)), sitk.GetArrayFromImage(
                sitk.ReadImage(labpath))
        # 强度标准化
        img_obj = np.asarray(
            (img_obj - img_obj.min()) / (img_obj.max() - img_obj.min()),
            dtype=np.float32)
        # slice标准化
        img_obj = (img_obj - img_obj.mean(axis=(1, 2), keepdims=True)
                   ) / img_obj.std(axis=(1, 2), keepdims=True)
        mask_obj = np.asarray(mask_obj, dtype=np.uint8)
        return torch.tensor(img_obj,
                            dtype=torch.float32).unsqueeze(1), torch.tensor(
                                mask_obj, dtype=torch.long).unsqueeze(1)


if __name__ == '__main__':
    import albumentations as A

    #

    # dataset = ACDCDataset2d("/home/yeep/project/py/ALSph2d/data/ACDCprecessed/train", transform=None)
    #
    # print(dataset[0][0])
    # print(dataset[0][1])
    # print()
    #
    # # dataset = ACDCDataset3d("/home/yeep/project/py/ALSph2d/data/ACDCprecessed/valid")
    # # print(dataset[0][0])
    # # print(dataset[0][1])
    # # print()

    dataset = ISICDataset("/home/yeep/project/py/AL-ACDC/data/ISIC")
    print(dataset[0])
