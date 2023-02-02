import horovod.torch as hvd
import logging
import numpy as np
import torch.multiprocessing as mp

from copy import deepcopy
from continuum import ClassIncremental, InstanceIncremental
from continuum.tasks import TaskSet
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import get_dataset
from preprocess import get_transform
from sampler import MyDistributedSampler


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
_CONTINUAL_ARGS = {
    "scenario",
    "increment",
    "initial_increment",
    "num_tasks",
    "concatenate_tasksets",
}
_TRANSFORM_ARGS = {"transform_name"}
_OTHER_ARGS = {
    "use_amp",
    "use_dali",
    "use_dali_cuda",
    "fp16_dali",
    "distributed",
    "shard"
}


class DataRegime(object):
    def __init__(self, hvd, config={}):
        self.hvd = hvd
        self.config = config
        self.epoch = 0
        self.task_id = 0
        self.steps = None
        self.tasksets = None
        self.total_num_classes = 0
        self.concat_taskset = None
        self.continual_test_taskset = []
        self.sampler = None
        self.loader = None
        self.classes_mask = None

        self.config = self.get_config(config)
        if self.config["others"].get("use_dali", False):
            try:
                global DaliDataLoader
                from data_loader_dali import DaliDataLoader
            except ImportError:
                raise ImportError(
                    "Please install NVIDIA DALI to run this app.")

        self.get_data(True)

    """Get the current taskset"""
    def get_data(self, force_update=False):
        if force_update:
            self._transform = get_transform(training=self.config["data"].get(
                "split", True) == "train", **self.config["transform"])
            self.config["data"].setdefault("transform", self._transform)
            self._data = self.get_taskset()
            logging.debug(
                f"DATA LOADER {self.config['data']['split']} - taskset updated: changed to {self.task_id}, len = {len(self._data)}"
            )

        return self._data

    def get_loader(self, force_update=False):
        if self.loader is None or force_update:
            # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
            # issues with Infiniband implementations that are not fork-safe
            if (
                self.config["loader"]["num_workers"] > 0
                and hasattr(mp, "_supports_context")
                and mp._supports_context
                and "forkserver" in mp.get_all_start_methods()
            ):
                self.config["loader"]["multiprocessing_context"] = "forkserver"

            if self.config["others"].get("use_dali", False):
                precision = 16 if self.config["others"].get(
                    "fp16_dali", False) else 32
                self.loader = DaliDataLoader(
                    self._data, self.task_id,
                    self.config["others"].get("use_dali_cuda", False),
                    device_id=hvd.local_rank(), shard_id=hvd.rank(),
                    num_shards=hvd.size(), precision=precision,
                    training=self.config["data"].get("split", True) == "train",
                    **self.config["loader"])
                logging.debug(
                    f"DATA LOADER {self.config['data']['split']} - data distributed using DALI")
            else:
                if self.config["others"].get("distributed", False):
                    if self.config["others"].get("shard", False):
                        self.config["loader"]["sampler"] = MyDistributedSampler(
                            self._data, num_replicas=hvd.size(), rank=hvd.rank()
                        )
                    else:
                        self.config["loader"]["sampler"] = DistributedSampler(
                            self._data, num_replicas=hvd.size(), rank=hvd.rank()
                        )
                    self.sampler = self.config["loader"]["sampler"]
                self.config["loader"]["shuffle"] = self.sampler is None
                self.loader = DataLoader(self._data, **self.config["loader"])
                logging.debug(
                    f"DATA LOADER {self.config['data']['split']} - data distributed using sampler")

        return self.loader

    """Get the taskset refered by self.task_id"""
    def get_taskset(self):
        if self.tasksets is None:
            self.prepare_tasksets()

        current_taskset = self.tasksets[self.task_id]
        if self.config["data"].get("split") == "train":
            if self.config["continual"].get("concatenate_tasksets", False):
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
        """
        else:
            if self.config['continual'].get('scenario', 'class'):
                self.continual_test_taskset.append(current_taskset)
                return self.continual_test_taskset
        """
        mask = np.zeros(self.total_num_classes)
        mask[current_taskset.get_classes()] = 1
        self.classes_mask = mask
        return current_taskset

    def prepare_tasksets(self):
        dataset = get_dataset(**self.config["data"])
        if self.config["data"].get("continual", False):
            continual_config = self.config["continual"]
            if continual_config.get("scenario") == "class":
                ii = continual_config.get("initial_increment")
                i = continual_config.get("increment")

                self.tasksets = ClassIncremental(
                    dataset,
                    initial_increment=ii,
                    increment=i,
                    transformations=[self.config["data"]["transform"]],
                )
            else:
                self.tasksets = InstanceIncremental(
                    dataset,
                    nb_tasks=continual_config.get("num_tasks"),
                    transformations=[self.config["data"]["transform"]],
                )
        else:
            self.tasksets = [
                dataset.to_taskset(
                    trsf=[self.config["data"]["transform"]]
                )
            ]
        logging.info(f"Prepared {len(self.tasksets)} tasksets")
        self.total_num_classes = len(dataset.list_classes)

    def set_epoch(self, epoch):
        logging.debug(
            f"DATA LOADER {self.config['data']['split']} - set epoch: {epoch}"
        )
        self.epoch = epoch
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def set_task_id(self, task_id):
        logging.debug(
            f"DATA LOADER {self.config['data']['split']} - set task id: {task_id}"
        )
        if self.task_id != task_id:
            self.task_id = task_id
            self.get_data(True)

    def __len__(self):
        return len(self._data) or 1

    def __str__(self):
        return str(self.config)

    def __repr__(self):
        return str(self.config)

    def get_config(self, config):
        loader_config = {k: v for k,
                         v in config.items() if k in _DATALOADER_ARGS}
        data_config = {k: v for k, v in config.items() if k in _DATA_ARGS}
        continual_config = {k: v for k,
                            v in config.items() if k in _CONTINUAL_ARGS}
        transform_config = {k: v for k,
                            v in config.items() if k in _TRANSFORM_ARGS}
        other_config = {k: v for k, v in config.items() if k in _OTHER_ARGS}

        transform_config.setdefault("transform_name", data_config["dataset"])

        return {
            "data": data_config,
            "loader": loader_config,
            "continual": continual_config,
            "transform": transform_config,
            "others": other_config,
        }

    def get(self, key, default=None):
        return self.config.get(key, default)
