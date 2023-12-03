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
    in_memory_shard=0,
    num_train_scans=1,
    num_in_memory_train_scans=1,
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
        logging.info(f"Assembling Ptycho {split} dataset...")

        H = 256
        W = 256
        # reshape all training images to this dimension
        im_shape = (256, 256)

        real_space = []
        diff_data = []
        amp_data = []
        ph_data = []

        # Train or validate workflow
        if split == "train" or split == "validate":
            # the square of the phase image must be above this value to be placed into training
            mean_phsqr_val = 0.02

            train_scans = list(range(204, 340 + 1))
            train_scans += list(range(457, 486 + 1))
            train_scans = train_scans[: min(num_train_scans, len(train_scans))]

            # shard training scans from the rank
            num_shards = hvd.size()
            shard_id = hvd.rank()
            shard_size = len(train_scans) // num_shards
            shard_offset = shard_size * shard_id
            train_scans = train_scans[shard_offset : shard_offset + shard_size]

            # inside the rank shard, take the current in-memory shard
            in_memory_shard_offset = num_in_memory_train_scans * in_memory_shard
            train_scans = train_scans[
                in_memory_shard_offset : min(
                    in_memory_shard_offset + num_in_memory_train_scans, len(train_scans)
                )
            ]

            for scan_num in tqdm(
                train_scans,
                desc="Loading ptychography scans",
            ):
                r_space = np.load(f"{root}/train/{scan_num}/patched_psi.npy")

                random.seed(42)
                num_samples_eval = int(0.1 * len(r_space))
                random_indices = random.sample(range(len(r_space)), num_samples_eval)
                if not train:
                    r_space = r_space[random_indices]
                else:
                    train_indices = []
                    for i in range(len(r_space)):
                        if i not in random_indices:
                            train_indices.append(i)
                    r_space = r_space[train_indices]

                # permutation_indices = np.random.permutation(len(r_space))
                # N_TRAIN = len(r_space)
                # N_VALID = int(0.2 * N_TRAIN)
                # if train:
                #    r_space = r_space[: N_TRAIN - N_VALID]
                # else:
                #    r_space = r_space[N_TRAIN - N_VALID : N_TRAIN]
                # r_space = r_space[permutation_indices]

                real_space.append(r_space)
                ampli = np.abs(r_space)
                amp_data.append(ampli)
                phase = np.angle(r_space)
                ph_data.append(phase)
                diff_data.append(
                    np.load(
                        f"{root}/train/{scan_num}/cropped_exp_diffr_data.npy"
                    )  # [permutation_indices]
                )

            logging.info("Converting the data to np array...")
            if len(diff_data) != 1:
                total_data_diff = np.concatenate(diff_data)
            else:
                _ = np.asarray(diff_data, dtype="float32")
                _ = _[0, :, :, :]
                total_data_diff = _[:, np.newaxis, :, :]
            total_data_amp = np.concatenate(amp_data)
            total_data_phase = np.concatenate(ph_data)

            logging.info("Removing useless data...")
            av_vals = np.mean(total_data_phase**2, axis=(1, 2))
            idx = np.argwhere(av_vals >= mean_phsqr_val)

            logging.info("Converting data to float32...")
            total_data_diff = total_data_diff[idx].astype("float32")
            total_data_amp = total_data_amp[idx].astype("float32")
            total_data_phase = total_data_phase[idx].astype("float32")

            logging.info("Reshaping data...")
            X_train = total_data_diff.reshape(-1, H, W)[:, np.newaxis, :, :]
            Y_I_train = total_data_amp.reshape(-1, H, W)[:, np.newaxis, :, :]
            Y_phi_train = total_data_phase.reshape(-1, H, W)[:, np.newaxis, :, :]

            logging.debug(f"Train data shape: {X_train.shape}")
            # X_train, Y_I_train, Y_phi_train = shuffle(
            #    X_train, Y_I_train, Y_phi_train, random_state=0
            # )

            # Training data
            X_train_tensor = torch.Tensor(X_train)
            Y_I_train_tensor = torch.Tensor(Y_I_train)
            Y_phi_train_tensor = torch.Tensor(Y_phi_train)
            logging.debug(
                f"""
                x shape: {X_train_tensor.shape}
                amp shape: {Y_I_train_tensor.shape}
                phi shape: {Y_phi_train_tensor.shape}
                """
            )

            """
            N_TRAIN = X_train_tensor.shape[0]
            N_VALID = int(0.2 * N_TRAIN)

            if train:
                return ReconstructionInMemoryDataset(
                    X_train_tensor[: N_TRAIN - N_VALID],
                    Y_I_train_tensor[: N_TRAIN - N_VALID],
                    Y_phi_train_tensor[: N_TRAIN - N_VALID],
                ), ["reconstruction"]
            else:
                return ReconstructionInMemoryDataset(
                    X_train_tensor[N_TRAIN - N_VALID : N_TRAIN],
                    Y_I_train_tensor[N_TRAIN - N_VALID : N_TRAIN],
                    Y_phi_train_tensor[N_TRAIN - N_VALID : N_TRAIN],
                ), ["reconstruction"]
            """
            return ReconstructionInMemoryDataset(
                X_train_tensor,
                Y_I_train_tensor,
                Y_phi_train_tensor,
            ), ["reconstruction"]
        else:
            start_scan = 489
            end_scan = 489

            for scan_num in tqdm(
                range(start_scan, end_scan + 1),
                desc="Loading ptychography scans",
            ):
                r_space = np.load(f"{root}/test/{scan_num}/patched_psi.npy")
                real_space.append(r_space)
                ampli = np.abs(r_space)
                amp_data.append(ampli)
                phase = np.angle(r_space)
                ph_data.append(phase)

                diff_data.append(
                    np.load(f"{root}/test/{scan_num}/cropped_exp_diffr_data.npy")
                )

            if len(diff_data) != 1:
                total_data_diff = np.concatenate(diff_data)
            else:
                _ = np.asarray(diff_data, dtype="float32")
                _ = _[0, :, :, :]
                total_data_diff = _[:, np.newaxis, :, :]

            logging.info("Converting the data to np array...")
            total_data_amp = np.concatenate(amp_data)
            total_data_phase = np.concatenate(ph_data)

            logging.info("Converting data to float32...")
            total_data_diff = total_data_diff.astype("float32")
            total_data_amp = total_data_amp.astype("float32")
            total_data_phase = total_data_phase.astype("float32")

            logging.info("Reshaping data...")
            X_test = total_data_diff.reshape(-1, H, W)[:, np.newaxis, :, :]
            Y_I_test = total_data_amp.reshape(-1, H, W)[:, np.newaxis, :, :]
            Y_phi_test = total_data_phase.reshape(-1, H, W)[:, np.newaxis, :, :]

            logging.debug(f"Test data shape: {X_test.shape}")

            # Testing data
            X_test_tensor = torch.Tensor(X_test)
            Y_I_test_tensor = torch.Tensor(Y_I_test)
            Y_phi_test_tensor = torch.Tensor(Y_phi_test)
            logging.debug(
                f"""
                x shape: {X_test_tensor.shape}
                amp shape: {Y_I_test_tensor.shape}
                phi shape: {Y_phi_test_tensor.shape}
                """
            )

            return ReconstructionInMemoryDataset(
                X_test_tensor,
                Y_I_test_tensor,
                Y_phi_test_tensor,
            ), ["reconstruction"]

    else:
        raise ValueError("Unknown dataset")


class ReconstructionInMemoryDataset(_ContinuumDataset):
    """Continuum dataset for in-memory data.

    :param x: Numpy array of images or paths to images for the train set.
    :param y_amp: Targets for the train set.
    :param data_type: Format of the data.
    :param t_train: Optional task ids for the train set.
    """

    def __init__(
        self,
        x: np.ndarray,
        y_amp: np.ndarray,
        y_ph: np.ndarray,
        t: Union[None, np.ndarray] = None,
        data_type: TaskType = TaskType.TENSOR,
    ):
        self._data_type = data_type
        super().__init__(download=False)

        if len(x) != len(y_amp) or len(x) != len(y_ph):
            raise ValueError(
                f"Number of datapoints ({len(x)}) != number of targets ({len(y_amp)}, {len(y_ph)})!"
            )
        if t is not None and len(t) != len(x):
            raise ValueError(
                f"Number of datapoints ({len(x)}) != number of task ids ({len(t)})!"
            )

        self.data = (x, y_amp, y_ph, t)
        self._nb_classes = 1

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
