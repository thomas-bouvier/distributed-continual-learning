import os
import torch

from continuum import datasets
from continuum.tasks import TaskType
from filelock import FileLock


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
            return datasets.MNIST(data_path=root, train=train, download=False)

    elif dataset == "cifar10":
        root = os.path.join(root, "CIFAR10")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.CIFAR10(data_path=root, train=train, download=False)

    elif dataset == "cifar100":
        root = os.path.join(root, "CIFAR100")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.CIFAR100(data_path=root, train=train, download=False)

    elif dataset == "tinyimagenet":
        root = os.path.join(root, "TinyImageNet")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.TinyImageNet200(data_path=root, train=train, download=False)

    elif dataset == "imagenet100small":
        root = os.path.join(root, "ImageNet100-small")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            data_subset = "train_100_small.txt" if train else "val_100_small.txt"
            return datasets.ImageNet100(
                data_path=root,
                train=train,
                data_subset=os.path.join(root, data_subset),
            )

    elif dataset == "imagenet100":
        root = os.path.join(root, "ImageNet100")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            data_subset = "train_100.txt" if train else "val_100.txt"
            return datasets.ImageNet100(
                data_path=root,
                train=train,
                data_subset=os.path.join(root, data_subset),
            )

    elif dataset == "imagenet":
        root = os.path.join(root, "ImageNet")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.ImageNet1000(data_path=root, train=train)

    elif dataset == "imagenet_blurred":
        root = os.path.join(root, "ImageNet_blurred")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.ImageNet1000(data_path=root, train=train)

    elif dataset == "core50":
        root = os.path.join(root, "CORe50")
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.Core50(data_path=root, train=train, download=False)

    else:
        raise ValueError("Unknown dataset")
