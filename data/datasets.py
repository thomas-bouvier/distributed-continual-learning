import horovod.torch as hvd
import logging
import numpy as np
import os
import random
import torch

from continuum import datasets
from continuum.datasets import _ContinuumDataset
from continuum.tasks import TaskType
from filelock import FileLock
from random import shuffle
from scipy.stats import circmean
from sklearn.utils import shuffle
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Union


COMPATIBILITY = ["class", "instance"]


def get_dataset(
    dataset="mnist",
    continual=False,
    split="train",
    transform=None,
    dataset_dir="./data",
):
    train = split == "train"
    root = os.path.expanduser(dataset_dir)

    if dataset == "mnist":
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return (
                datasets.MNIST(data_path=root, train=train, download=True),
                COMPATIBILITY,
            )

    elif dataset == "cifar10":
        root = os.path.join(root, "CIFAR10")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return (
                datasets.CIFAR10(data_path=root, train=train, download=True),
                COMPATIBILITY,
            )

    elif dataset == "cifar100":
        root = os.path.join(root, "CIFAR100")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return (
                datasets.CIFAR100(data_path=root, train=train, download=False),
                COMPATIBILITY,
            )

    elif dataset == "tinyimagenet":
        root = os.path.join(root, "TinyImageNet")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return (
                datasets.TinyImageNet200(data_path=root, train=train, download=False),
                COMPATIBILITY,
            )

    elif dataset == "imagenet100small":
        root = os.path.join(root, "ImageNet100-small")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            data_subset = "train_100_small.txt" if train else "val_100_small.txt"
            return (
                datasets.ImageNet100(
                    data_path=root,
                    train=train,
                    data_subset=os.path.join(root, data_subset),
                ),
                COMPATIBILITY,
            )

    elif dataset == "imagenet100":
        root = os.path.join(root, "ImageNet100")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            data_subset = "train_100.txt" if train else "val_100.txt"
            return (
                datasets.ImageNet100(
                    data_path=root,
                    train=train,
                    data_subset=os.path.join(root, data_subset),
                ),
                COMPATIBILITY,
            )

    elif dataset == "imagenet":
        root = os.path.join(root, "ImageNet")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.ImageNet1000(data_path=root, train=train), COMPATIBILITY

    elif dataset == "imagenet_blurred":
        root = os.path.join(root, "ImageNet_blurred")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.ImageNet1000(data_path=root, train=train), COMPATIBILITY

    elif dataset == "core50":
        root = os.path.join(root, "CORe50")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return (
                datasets.Core50(data_path=root, train=train, download=False),
                COMPATIBILITY,
            )

    elif dataset == "ptycho":
        root = os.path.join(root, "Ptycho")

        num_train_scans = 156
        train_scans = list(range(204, 340 + 1))
        train_scans += list(range(457, 486 + 1))
        train_scans = train_scans[: min(num_train_scans, len(train_scans))]

        train_scans_paths = np.array([f"{root}/train/{scan}" for scan in train_scans])

        return DiffractionDataset(train_scans_paths), ["reconstruction"]
    else:
        raise ValueError("Unknown dataset")


class DiffractionDataset(_ContinuumDataset):
    """Continuum dataset for diffraction data.

    :param x: Numpy array of paths to diffractions for the train set.
    :param data_type: Format of the data.
    :param t_train: Optional task ids for the train set.
    """

    def __init__(self, x: np.ndarray, t: Union[None, np.ndarray] = None):
        self._data_type = TaskType.IMAGE_PATH
        super().__init__(download=False)

        self.data = (x, t)
        self._nb_classes = 1

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.data

    @property
    def nb_classes(self) -> List[int]:
        return self._nb_classes

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    @data_type.setter
    def data_type(self, data_type: TaskType) -> None:
        self._data_type = data_type
