import glob
import logging
import numpy as np
import os
import torch
import random

from continuum import datasets
from continuum.datasets import InMemoryDataset
from continuum.tasks import TaskType
from filelock import FileLock
from scipy.stats import circmean
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
                datasets.MNIST(data_path=root, train=train, download=False),
                COMPATIBILITY,
            )

    elif dataset == "cifar10":
        root = os.path.join(root, "CIFAR10")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return (
                datasets.CIFAR10(data_path=root, train=train, download=False),
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
        logging.info("Assembling Ptycho dataset...")
        datafiles_train = glob.glob(root + "/trainingData/*.npz")
        diffraction_downscaling = 1

        x = []
        y = []
        positions_train = []
        for df in datafiles_train:
            logging.info(df)
            with np.load(df) as f:
                try:
                    x.append(np.array(f["reciprocal"]))
                except Exception as e:
                    logging.info(e)
                else:
                    positions_train.append(f["position"])
                    realspace = np.array(f["real"])
                    phases = np.angle(realspace)
                    phase_mean = circmean(phases, low=-np.pi, high=np.pi)
                    y.append(realspace * np.exp(-1j * phase_mean))

        logging.debug(
            f"Shape of new training data is:\n"
            f"\tx: {np.shape(x)}\n"
            f"\ty: {np.shape(y)}"
        )

        shape12 = np.array(x).shape[-2:]
        x = np.reshape(x, [-1, *shape12])
        y = np.reshape(y, [-1, *shape12])
        y_ph = np.angle(y)

        logging.debug(f"Before downscaling, max of x is {np.max(x)}")
        x = np.floor(x / diffraction_downscaling)
        logging.debug(f"After downscaling, max of x is {np.max(x)}")

        x = x[:, np.newaxis, ...]  # .astype("float32")
        y_ph = y_ph[:, np.newaxis, ...]

        logging.debug(
            f"Shape of new training data is:\n"
            f"\tx: {np.shape(x)}\n"
            f"\ty_ph: {np.shape(y_ph)}"
        )

        ntrain_full = np.shape(x)[0]
        valid_data_ratio = 0.1
        nvalid = int(ntrain_full * valid_data_ratio)
        ntrain = ntrain_full - nvalid
        indices = list(range(ntrain_full))
        np.random.seed(0)
        np.random.shuffle(indices)
        x = x[indices]
        y_ph = y_ph[indices]

        if train:
            return ReconstructionInMemoryDataset(x[:ntrain], y_ph[:ntrain]), [
                "reconstruction"
            ]
        else:
            return ReconstructionInMemoryDataset(x[ntrain:], y_ph[ntrain:]), [
                "reconstruction"
            ]

    else:
        raise ValueError("Unknown dataset")


class ReconstructionInMemoryDataset(InMemoryDataset):
    """Continuum dataset for in-memory data.

    :param x: Numpy array of images or paths to images for the train set.
    :param y: Targets for the train set.
    :param data_type: Format of the data.
    :param t_train: Optional task ids for the train set.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: Union[None, np.ndarray] = None,
        data_type: TaskType = TaskType.TENSOR,
        train: bool = True,
        download: bool = True,
    ):
        self._data_type = data_type
        super().__init__(x, y, train=train, download=download)

        if len(x) != len(y):
            raise ValueError(
                f"Number of datapoints ({len(x)}) != number of labels ({len(y)})!"
            )
        if t is not None and len(t) != len(x):
            raise ValueError(
                f"Number of datapoints ({len(x)}) != number of task ids ({len(t)})!"
            )

        self.data = (x, y, t)
        self._nb_classes = 1

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
