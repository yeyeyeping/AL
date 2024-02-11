import yaml
import os
import torch
import scipy.ndimage as ndimage
import numpy as np
import albumentations as A
from os.path import join
import random
from argparse import ArgumentParser
import time
import shutil
import logging
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from collections import Sequence


class SubsetSampler(SubsetRandomSampler):

    def __iter__(self):
        random.shuffle(self.indices)
        return iter(self.indices)


def save_query_plot(folder, labeled_percent, dice_list):
    import matplotlib.pyplot as plt
    with open(f"{folder}/result.txt", "w") as fp:
        fp.write("x:")
        fp.write(str(labeled_percent))
        fp.write("\ny:")
        fp.write(str(dice_list))
    plt.plot(labeled_percent, dice_list)
    plt.savefig(f"{folder}/result.jpg")


def get_samplers(data_num, initial_labeled, g):
    initial_labeled = int(data_num * initial_labeled)
    data_indice = list(range(data_num))
    np.random.shuffle(data_indice)
    return SubsetSampler(data_indice[:initial_labeled],
                         generator=g), SubsetSampler(
                             data_indice[initial_labeled:], generator=g)


def get_largest_k_components(image, k=1):
    """
    Get the largest K components from 2D or 3D binary image.

    :param image: The input ND array for binary segmentation.
    :param k: (int) The value of k.

    :return: An output array with only the largest K components of the input.
    """
    dim = len(image.shape)
    if (image.sum() == 0):
        print('the largest component is null')
        return image
    if (dim < 2 or dim > 3):
        raise ValueError("the dimension number should be 2 or 3")
    s = ndimage.generate_binary_structure(dim, 1)
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    sizes_sort = sorted(sizes, reverse=True)
    kmin = min(k, numpatches)
    output = np.zeros_like(image)
    for i in range(kmin):
        labeli = np.where(sizes == sizes_sort[i])[0] + 1
        output = output + np.asarray(labeled_array == labeli, np.uint8)
    return output


def label_smooth(volume):
    [D, H, W] = volume.shape
    s = ndimage.generate_binary_structure(2, 1)
    for d in range(D):
        if (volume[d].sum() > 0):
            volume_d = get_largest_k_components(volume[d], k=5)
            if (volume_d.sum() < 10):
                volume[d] = np.zeros_like(volume[d])
                continue
            volume_d = ndimage.morphology.binary_closing(volume_d, s)
            volume_d = ndimage.morphology.binary_opening(volume_d, s)
            volume[d] = volume_d
    return volume


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader_Cadis(config):
    g = torch.Generator()
    g.manual_seed(config["Training"]["seed"])
    data_dir = config["Dataset"]["data_dir"]
    batch_size = config["Dataset"]["batch_size"]
    num_worker = config["Dataset"]["num_workers"]
    input_size = config["Dataset"]["input_size"]
    from dataset.ACDCDataset import ISICDataset

    train_transform = A.Compose([
        A.Compose([
            A.PadIfNeeded(384, 384),
            A.ShiftScaleRotate(rotate_limit=90),
            A.RandomCrop(*input_size)
        ],
            p=0.5),
        A.VerticalFlip(),
        A.HorizontalFlip(),
        A.RandomGamma(gamma_limit=(0.7, 1.5)),
        A.ColorJitter(brightness=32 / 255, saturation=0.2, hue=0.1),
        A.Normalize(max_pixel_value=1),
    ])
    #    A.Resize(256, 256),
    # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
    # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
    # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # train_transform = A.Compose([
    #     A.Compose([
    #         A.PadIfNeeded(384, 384),
    #         A.ShiftScaleRotate(rotate_limit=90),
    #         A.RandomCrop(*input_size)
    #     ],
    #               p=0.5),
    #     A.RandomGamma(gamma_limit=(0.7, 1.5)),
    #     A.ColorJitter(brightness=32 / 255, saturation=0.2, hue=0.1),
    #     A.Blur(blur_limit=3),
    #     A.OpticalDistortion(),
    #     A.VerticalFlip(),
    #     A.HorizontalFlip(),
    #     A.Normalize(max_pixel_value=1)
    # ])
    test_transform = A.Compose([
        A.Normalize(max_pixel_value=1),
    ])

    dataset_train, dataset_val = ISICDataset(trainfolder=join(data_dir, "train"),
                                             transform=train_transform), \
        ISICDataset(trainfolder=join(data_dir, "val"),
                    transform=test_transform)

    labeled_sampler, unlabeled_sampler = get_samplers(
        len(dataset_train), config["AL"]["initial_labeled"], g=g)

    dulabeled = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            sampler=unlabeled_sampler,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            prefetch_factor=num_worker,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            num_workers=num_worker)

    dlabeled = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=batch_size,
                                           sampler=labeled_sampler,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           worker_init_fn=seed_worker,
                                           generator=g,
                                           prefetch_factor=num_worker,
                                           num_workers=num_worker)

    dval = torch.utils.data.DataLoader(dataset_val,
                                       batch_size=batch_size,
                                       persistent_workers=True,
                                       pin_memory=True,
                                       prefetch_factor=num_worker,
                                       worker_init_fn=seed_worker,
                                       generator=g,
                                       num_workers=num_worker)

    return {"labeled": dlabeled, "unlabeled": dulabeled, "valid": dval}


def get_dataloader_ISIC(config):
    g = torch.Generator()
    g.manual_seed(config["Training"]["seed"])
    data_dir = config["Dataset"]["data_dir"]
    batch_size = config["Dataset"]["batch_size"]
    num_worker = config["Dataset"]["num_workers"]
    from dataset.ACDCDataset import ISICDataset
    train_transform = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.2),
        A.ColorJitter(brightness=32 / 255, saturation=0.5),
        A.Normalize(max_pixel_value=1),
    ])
    test_transform = A.Compose([
        A.Normalize(max_pixel_value=1),
    ])

    dataset_train, dataset_val = ISICDataset(trainfolder=join(data_dir, "train"),
                                             transform=train_transform), \
        ISICDataset(trainfolder=join(data_dir, "val"),
                    transform=test_transform)

    labeled_sampler, unlabeled_sampler = get_samplers(
        len(dataset_train), config["AL"]["initial_labeled"], g=g)

    dulabeled = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            sampler=unlabeled_sampler,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            prefetch_factor=num_worker,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            num_workers=num_worker)

    dlabeled = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=batch_size,
                                           sampler=labeled_sampler,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           worker_init_fn=seed_worker,
                                           generator=g,
                                           prefetch_factor=num_worker,
                                           num_workers=num_worker)

    dval = torch.utils.data.DataLoader(dataset_val,
                                       batch_size=batch_size,
                                       persistent_workers=True,
                                       pin_memory=True,
                                       prefetch_factor=num_worker,
                                       worker_init_fn=seed_worker,
                                       generator=g,
                                       num_workers=num_worker)

    return {"labeled": dlabeled, "unlabeled": dulabeled, "valid": dval}


def get_dataloader_ACDC(config, with_pseudo=False):
    g = torch.Generator()
    g.manual_seed(config["Training"]["seed"])
    data_dir = config["Dataset"]["data_dir"]
    batch_size = config["Dataset"]["batch_size"]
    num_worker = config["Dataset"]["num_workers"]
    from dataset.ACDCDataset import ACDCDataset2d, ACDCDataset3d
    train_transform = A.Compose([
        A.PadIfNeeded(256, 256),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(p=0.2),
        A.RandomCrop(192, 192),
        A.GaussNoise(0.005, 0, per_channel=False),
    ])
    dataset_train, dataset_val = ACDCDataset2d(trainfolder=join(data_dir, "train"),
                                               transform=train_transform), \
        ACDCDataset3d(folder=join(data_dir, "valid"))
    labeled_sampler, *unlabeled_sampler = get_samplers(
        len(dataset_train),
        config["AL"]["initial_labeled"],
        with_pseudo=with_pseudo)

    dulabeled = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            sampler=unlabeled_sampler,
                                            persistent_workers=True,
                                            pin_memory=True,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            prefetch_factor=num_worker,
                                            num_workers=num_worker)

    dlabeled = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=batch_size,
                                           sampler=labeled_sampler,
                                           persistent_workers=True,
                                           pin_memory=True,
                                           worker_init_fn=seed_worker,
                                           generator=g,
                                           prefetch_factor=num_worker,
                                           num_workers=num_worker)

    dval = torch.utils.data.DataLoader(dataset_val,
                                       batch_size=1,
                                       persistent_workers=True,
                                       pin_memory=True,
                                       worker_init_fn=seed_worker,
                                       generator=g,
                                       prefetch_factor=num_worker,
                                       num_workers=num_worker)

    return {"labeled": dlabeled, "unlabeled": dulabeled, "valid": dval}


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * sigmoid_rampup(epoch, 80)


def read_yml(filepath):
    assert os.path.exists(filepath), "file not exist"
    with open(filepath) as fp:
        config = yaml.load(fp, yaml.FullLoader)
    return config


def random_seed(config):
    import torch.backends.cudnn as cudnn
    seed = config["Training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def init_logger(config):
    import sys
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    outputdir = config["Training"]["output_dir"]
    fh = logging.FileHandler(f"{outputdir}/{time.time()}.log")
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(f"Query Strategy: {outputdir}")
    logger.info(config)
    return logger


def parse_config():
    parser = ArgumentParser()
    parser.add_argument("--config",
                        "-c",
                        type=str,
                        default="config-ssl/config-ssl.yml")
    parser.add_argument("--strategy",
                        "-s",
                        type=str,
                        default="config-ssl/strategy.yml")
    args = parser.parse_args()

    config, all_strategy = read_yml(args.config), read_yml(args.strategy)

    config["Training"]["output_dir"] = config["AL"]["query_strategy"] if config["Training"]["output_dir"] is None or \
        config["Training"]["output_dir"] == "" \
        else config["Training"]["output_dir"]
    os.makedirs(config["Training"]["output_dir"], exist_ok=True)
    shutil.copy(args.config,
                join(config["Training"]["output_dir"], "config.yml"))

    config["Training"]["checkpoint_dir"] = os.path.join(
        config["Training"]["output_dir"], "checkpoint")
    os.makedirs(config["Training"]["checkpoint_dir"], exist_ok=True)
    config["all_strategy"] = all_strategy
    return config


def mode_filter(image, kernel_size=(16, 16)):
    '''
    image: h w 
    '''
    assert image.ndim == 2, f"{image.shape} dim must be 2"
    if type(kernel_size) == int:
        h = w = kernel_size
    elif type(kernel_size) is np.ndarray and len(kernel_size) == 2:
        kernel_size = kernel_size.astype(np.uint8)
        w, h = kernel_size
    else:
        raise Exception(f"dim of {kernel_size} is 1 or 2")

    o_h, o_w = image.shape[0] // h, image.shape[1] // w

    if o_h*h > image.shape[0]:
        o_h = o_h + 1

    if o_w*w > image.shape[1]:
        o_w = o_w + 1

    mode_arr = np.empty(shape=(o_h, o_w), dtype=np.uint8)
    i = 0
    for hi in range(0, image.shape[0], h):
        j = 0
        for wi in range(0, image.shape[1], w):
            wh, ww = hi + h, wi + w
            wh = wh if wh < image.shape[0] else image.shape[0]
            ww = ww if ww < image.shape[1] else image.shape[1]
            window = image[hi:wh, wi:ww]
            c = np.bincount(window.flatten())
            mode_arr[i, j] = np.argmax(c)
            j += 1
        i += 1
    return mode_arr
