import numpy as np
import os
import pandas as pd
import torch

from continuum import datasets
from continuum.datasets.base import _ContinuumDataset
from continuum.tasks import TaskType
from filelock import FileLock
from sklearn.preprocessing import MaxAbsScaler
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple


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

    elif dataset == "candle":
        root = os.path.join(root, "CANDLE")

        if not os.path.exists(os.path.join(root, "train/x.pt")) or not os.path.exists(
            os.path.join(root, "val/x.pt")
        ):
            train_path = os.path.join(root, "nt_train2.csv")
            test_path = os.path.join(root, "nt_test2.csv")
            df_train = (pd.read_csv(
                train_path, header=None).values).astype("float32")
            df_test = (pd.read_csv(test_path, header=None).values).astype(
                "float32")

            seqlen = df_train.shape[1]
            y_train = df_train[:, 0].astype("int")
            y_test = df_test[:, 0].astype("int")
            df_x_train = df_train[:, 1:seqlen].astype(np.float32)
            df_x_test = df_test[:, 1:seqlen].astype(np.float32)
            x_train = df_x_train
            x_test = df_x_test

            scaler = MaxAbsScaler()
            mat = np.concatenate((x_train, x_test), axis=0)
            mat = scaler.fit_transform(mat)

            x_train = mat[: x_train.shape[0], :]
            x_test = mat[x_train.shape[0]:, :]
            x_train = np.expand_dims(x_train, axis=1)
            x_test = np.expand_dims(x_test, axis=1)
            x_train = torch.as_tensor(x_train)
            x_test = torch.as_tensor(x_test)
            y_train = torch.as_tensor(y_train)
            y_test = torch.as_tensor(y_test)

            torch.save(x_test, os.path.join(root, "x_test.pt"))
            torch.save(y_test, os.path.join(root, "y_test.pt"))
            torch.save(x_train, os.path.join(root, "x_train.pt"))
            torch.save(y_train, os.path.join(root, "y_train.pt"))
            data_train = torch.load(os.path.join(root, "x_train.pt"))
            target_train = torch.load(os.path.join(root, "y_train.pt"))
            data_test = torch.load(os.path.join(root, "x_test.pt"))
            target_test = torch.load(os.path.join(root, "y_test.pt"))
            torch.save(data_train, os.path.join(root, "train/x.pt"))
            torch.save(target_train, os.path.join(root, "train/y.pt"))
            torch.save(data_test, os.path.join(root, "val/x.pt"))
            torch.save(target_test, os.path.join(root, "val/y.pt"))

        if train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "val")
        if continual:
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return CANDLE(
                    root=root,
                    train=train,
                    transform=transform,
                    target_transform=target_transform,
                    download=download,
                )
        else:
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return VIS_CANDLE(
                    root=root,
                    train=train,
                    transform=transform,
                    target_transform=target_transform,
                    download=download,
                )
    else:
        raise ValueError("Unknown dataset")


class CANDLE(_ContinuumDataset):
    def __init__(
        self, root: str = "", train: bool = True, download: bool = True, **kwargs
    ):
        super().__init__(data_path=root, train=train, download=download)
        self.dataset_type = VIS_CANDLE
        self.dataset = self.dataset_type(
            self.data_path, download=self.download, train=self.train, **kwargs
        )

    def get_data(self):
        return self.dataset.data, self.dataset.targets, None

    @property
    def transformations(self):
        return None

    @property
    def data_type(self):
        return TaskType.TEXT


class VIS_CANDLE(VisionDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(VIS_CANDLE, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.data = torch.load(os.path.join(root, "x.pt"))
        self.targets = torch.load(os.path.join(root, "y.pt"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def transformations(self):
        return None

    @property
    def data_type(self):
        return TaskType.TEXT