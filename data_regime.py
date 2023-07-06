import horovod.torch as hvd
import logging
import numpy as np
import torch.multiprocessing as mp

from copy import deepcopy
from continuum import ClassIncremental, InstanceIncremental
from continuum.tasks import TaskSet
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.datasets import get_dataset
from data.preprocess import get_transform
from data.scenarios import ReconstructionInstanceIncremental


_DATA_ARGS = {
    "dataset",
    "split",
    "transform",
    "target_transform",
    "download",
    "dataset_dir",
    "continual",
}
_DATALOADER_ARGS = {
    "batch_size",
    "sampler",
    "batch_sampler",
    "num_workers",
    "collate_fn",
    "pin_memory",
    "drop_last",
    "timeout",
    "worker_init_fn",
}
_TASKS_ARGS = {
    "scenario",
    "increment",
    "initial_increment",
    "num_tasks",
    "concatenate_tasksets",
}
_TRANSFORM_ARGS = {"transform_name"}

# TODO: plug this class onto augmented mini-batches in Neomem


class DataRegime:
    def __init__(self, hvd, config={}):
        self.hvd = hvd
        self.config = config
        self.epoch = 0
        self.task_id = -1
        self.tasksets = None
        self.total_num_classes = 0
        self.total_num_samples = 0
        self.concat_taskset = None
        self.continual_test_taskset = []
        self.sampler = None
        self.loader = None
        self.sample_shape = None
        self.data_len = 0
        self.classes_mask = None

        self.previous_classes_mask = None
        self.previous_loaders = {}
        self.config = self.get_config(config)

        self.use_dali = True
        try:
            global DaliDataLoader
            from data.load import DaliDataLoader
        except ImportError:
            logging.info(
                f"NVIDIA DALI is not installed, fallback to the native PyTorch"
                " native dataloader."
            )
            self.use_dali = False

        self.prepare_tasksets()
        self.get_data()

    def prepare_tasksets(self):
        dataset, compatibility = get_dataset(**self.config["data"])
        self.total_num_samples = len(dataset.get_data()[0])
        scenario = self.config["tasks"].get("scenario", None)
        if scenario:
            assert (
                scenario in compatibility
            ), f"{self.config['data']['dataset']} is only compatible with {compatibility} scenarios"

        if scenario == "class":
            ii = self.config["tasks"].get("initial_increment", 0)
            i = self.config["tasks"].get("increment", 1)

            self.tasksets = ClassIncremental(
                dataset,
                initial_increment=ii,
                increment=i,
                transformations=[self.config["transform"]["compose"]],
            )
        elif scenario == "instance":
            self.tasksets = InstanceIncremental(
                dataset,
                nb_tasks=self.config["tasks"].get("num_tasks", 5),
                transformations=[self.config["transform"]["compose"]],
            )
        else:
            self.tasksets = InstanceIncremental(
                dataset,
                nb_tasks=1,
                transformations=[self.config["transform"]["compose"]],
            )
        logging.info(
            f"Prepared {len(self.tasksets)} {self.config['data']['split']} tasksets"
        )
        self.total_num_classes = self.tasksets.nb_classes

    def get_taskset(self):
        """
        Get the taskset refered to by self.task_id, with all previous data
        accumulated if enabled by parameter `concatenate_tasksets`.
        """
        current_taskset = self.tasksets[self.task_id]

        if self.config["data"].get("split") == "train":
            if self.config["tasks"].get("concatenate_tasksets", False):
                if self.concat_taskset is None:
                    self.concat_taskset = current_taskset
                else:
                    logging.debug(
                        f"DATA LOADER {self.config['data']['split']} - concatenating taskset with all previous ones.."
                    )
                    x, y, t = self.concat_taskset.get_raw_samples(
                        np.arange(len(self.concat_taskset))
                    )
                    nx, ny, nt = current_taskset.get_raw_samples(
                        np.arange(len(current_taskset))
                    )
                    x = np.concatenate((x, nx))
                    y = np.concatenate((y, ny))
                    t = np.concatenate((t, nt))
                    self.concat_taskset = TaskSet(
                        x,
                        y,
                        t,
                        trsf=current_taskset.trsf,
                        data_type=current_taskset.data_type,
                    )
                return self.concat_taskset

        # Update the mask of observed classes so far
        mask = np.zeros(self.total_num_classes, dtype=bool)
        mask[current_taskset.get_classes()] = True
        if self.previous_classes_mask is None:
            self.previous_classes_mask = mask.copy()
        self.previous_classes_mask[mask] = True
        self.classes_mask = mask

        return current_taskset

    def get_data(self):
        """Get the current taskset"""
        data = self.get_taskset()
        logging.debug(
            f"DATA LOADER {self.config['data']['split']} - taskset updated: changed to {self.task_id}, len = {len(data)}"
        )

        return data

    def get_loader(self, task_id):
        """
        Get the loader refered to by self.task_id
        """

        if task_id == self.task_id:
            return self.loader

        if task_id in self.previous_loaders.keys():
            self.data_len = self.previous_loaders[task_id][1]
            loader = self.previous_loaders[task_id][0]
            self.sample_shape = next(iter(loader))[0][0].size()
            return loader

        logging.debug(
            f"DATA LOADER {self.config['data']['split']} - set task id: {task_id}"
        )
        self.task_id = task_id

        data = self.get_data()
        self.data_len = len(data)

        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (
            self.config["loader"]["num_workers"] > 0
            and hasattr(mp, "_supports_context")
            and mp._supports_context
            and "forkserver" in mp.get_all_start_methods()
        ):
            self.config["loader"]["multiprocessing_context"] = "forkserver"

        if self.use_dali:
            # If the current data regime is used for training, we can deallocate
            # the current data loader
            if self.config["data"].get("split") == "train" and self.loader is not None:
                self.loader.release()

            precision = 16 if False else 32
            self.loader = DaliDataLoader(
                data,
                self.task_id,
                device_id=hvd.local_rank(),
                shard_id=hvd.rank(),
                num_shards=hvd.size(),
                precision=precision,
                training=self.config["data"].get("split", True) == "train",
                **self.config["loader"],
            )
            logging.debug(
                f"DATA LOADER {self.config['data']['split']} - data distributed using DALI"
            )

        else:
            if hvd.size() > 1:
                self.config["loader"]["sampler"] = DistributedSampler(
                    data, num_replicas=hvd.size(), rank=hvd.rank()
                )
                self.sampler = self.config["loader"]["sampler"]
            self.config["loader"]["shuffle"] = self.sampler is None
            self.loader = DataLoader(data, **self.config["loader"])
            logging.debug(
                f"DATA LOADER {self.config['data']['split']} - data distributed using sampler"
            )

        if self.config["data"].get("split") == "validate":
            self.previous_loaders[self.task_id] = (self.loader, self.data_len)

        self.sample_shape = next(iter(self.loader))[0][0].size()
        return self.loader

    def set_epoch(self, epoch):
        logging.debug(
            f"DATA LOADER {self.config['data']['split']} - set epoch: {epoch}"
        )
        self.epoch = epoch
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def get_config(self, config):
        loader_config = {k: v for k, v in config.items() if k in _DATALOADER_ARGS}
        data_config = {k: v for k, v in config.items() if k in _DATA_ARGS}
        tasks_config = {k: v for k, v in config.items() if k in _TASKS_ARGS}
        transform_config = {k: v for k, v in config.items() if k in _TRANSFORM_ARGS}

        transform_config.setdefault("transform_name", data_config["dataset"])
        compose = get_transform(
            training=data_config.get("split", True) == "train",
            **transform_config,
        )
        transform_config.setdefault("compose", compose)

        return {
            "data": data_config,
            "loader": loader_config,
            "tasks": tasks_config,
            "transform": transform_config,
        }

    def get(self, key, default=None):
        return self.config.get(key, default)
