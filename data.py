import horovod.torch as hvd
import numpy as np
import os
import torch

from copy import deepcopy
from continuum import datasets as datasets_c
from continuum import ClassIncremental, InstanceIncremental
from continuum.tasks import TaskSet
from filelock import FileLock
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets

from preprocess import get_transform
from regime import Regime      
from sampler import MyDistributedSampler


def get_dataset(dataset='mnist', continual=False, split='train', transform=None,
        target_transform=None, download=False, dataset_dir='./data'):
    train = (split == 'train')
    root = os.path.expanduser(dataset_dir)

    if dataset == 'mnist':
        if continual:
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return datasets_c.MNIST(data_path=root,
                                train=train,
                                download=download)
        else:
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return datasets.MNIST(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)

    elif dataset == 'cifar10':
        if continual:
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return datasets_c.CIFAR10(data_path=os.path.join(root, 'CIFAR10'),
                                train=train,
                                download=download)
        else:
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return datasets.CIFAR10(root=os.path.join(root, 'CIFAR10'),
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)

    elif dataset == 'cifar100':
        if continual:
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return datasets_c.CIFAR100(data_path=os.path.join(root, 'CIFAR100'),
                                train=train,
                                download=download)
        else:
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return datasets.CIFAR100(root=os.path.join(root, 'CIFAR100'),
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)

    elif dataset == 'tinyimagenet':
        if continual:
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return datasets_c.TinyImageNet200(data_path=os.path.join(root, 'TinyImageNet'),
                                train=train,
                                download=download)
        else:
            root = os.path.join(root, 'TinyImageNet')
            if train:
                root = os.path.join(root, 'tiny-imagenet-200/train')
            else:
                root = os.path.join(root, 'tiny-imagenet-200/val')
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                return datasets.ImageFolder(root=root,
                                            transform=transform,
                                            target_transform=target_transform)

    elif dataset == 'imagenet':
        root = os.path.join(root, 'ImageNet')
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.ImageFolder(root=root,
                                        transform=transform,
                                        target_transform=target_transform)

    elif dataset == 'imagenet_blurred':
        root = os.path.join(root, 'ImageNet_blurred')
        if train:
            root = os.path.join(root, 'train_blurred')
        else:
            root = os.path.join(root, 'val_blurred')
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.ImageFolder(root=root,
                                        transform=transform,
                                        target_transform=target_transform)


_DATA_ARGS = {'dataset', 'split', 'transform', 'target_transform', 'download',
                    'dataset_dir', 'continual'}
_DATALOADER_ARGS = {'batch_size', 'shuffle', 'sampler', 'batch_sampler',
                    'num_workers', 'collate_fn', 'pin_memory', 'drop_last',
                    'timeout', 'worker_init_fn'}
_CONTINUAL_ARGS = {'scenario', 'increment', 'initial_increment', 'num_tasks',
                    'concatenate_tasksets'}
_TRANSFORM_ARGS = {'transform_name'}
_OTHER_ARGS = {'distributed', 'shard'}


"""
Inspired by https://github.com/eladhoffer/convNet.pytorch/blob/master/data.py
"""
class DataRegime(object):
    def __init__(self, hvd, regime, defaults={}):
        self.regime = Regime(regime, deepcopy(defaults))
        self.hvd = hvd
        self.epoch = 0
        self.task_id = 0
        self.current_task_id = 0
        self.steps = None
        self.tasksets = None
        self.concat_taskset = None
        self.get_loader(True)


    def get_loader(self, force_update=False):
        if force_update or self.regime.update(self.epoch, self.steps) or self.task_id > self.current_task_id:
            self.current_task_id = self.task_id
            config = self.get_config()

            self._transform = get_transform(**config['transform'])
            config['data'].setdefault('transform', self._transform)
            self._data = self.get_taskset(config)

            if config['others'].get('distributed', False):
                config['loader']['sampler'] = DistributedSampler(
                    self._data, num_replicas=hvd.size(), rank=hvd.rank()
                )

            config['loader']['shuffle'] = None

            self._sampler = config['loader'].get('sampler', None)
            self._loader = DataLoader(self._data, **config['loader'])

        return self._loader
    

    def get_taskset(self, config):
        if config['data'].get('continual', False):
            if self.tasksets is None:
                self.prepare_tasksets(config)
            
            current_taskset = self.tasksets[self.current_task_id]
            if config['continual'].get('concatenate_tasksets', False):
                if self.concat_taskset is None:
                    self.concat_taskset = current_taskset
                else:
                    x, y, t = self.concat_taskset.get_raw_samples(np.arange(len(self.concat_taskset)))
                    nx, ny, nt = current_taskset.get_raw_samples(np.arange(len(current_taskset)))
                    x = np.concatenate((x, nx))
                    y = np.concatenate((y, ny))
                    t = np.concatenate((t, nt))
                    self.concat_taskset = TaskSet(x, y, t, trsf=current_taskset.trsf, data_type=current_taskset.data_type)
                return self.concat_taskset

            return current_taskset

        return get_dataset(**config['data'])


    def prepare_tasksets(self, config):
        if config['continual'].get('scenario') == 'class':
            ii = config['continual'].get('initial_increment')
            i = config['continual'].get('increment')

            self.tasksets = ClassIncremental(
                get_dataset(**config['data']),
                initial_increment = ii,
                increment = i
            )
        else:
            self.tasksets = InstanceIncremental(
                get_dataset(**config['data']),
                nb_tasks = continual_config.get('num_tasks')
            )


    def set_epoch(self, epoch):
        self.epoch = epoch
        if self._sampler is not None and hasattr(self._sampler, 'set_epoch'):
            self._sampler.set_epoch(epoch)


    def set_task_id(self, task_id):
        self.task_id = task_id


    def __len__(self):
        return len(self._data) or 1


    def __repr__(self):
        return str(self.regime)


    def get_config(self):
        config = self.regime.config
        loader_config = {
            k: v for k, v in config.items() if k in _DATALOADER_ARGS}
        data_config = {
            k: v for k, v in config.items() if k in _DATA_ARGS}
        continual_config = {
            k: v for k, v in config.items() if k in _CONTINUAL_ARGS}
        transform_config = {
            k: v for k, v in config.items() if k in _TRANSFORM_ARGS}
        other_config = {
            k: v for k, v in config.items() if k in _OTHER_ARGS}

        transform_config.setdefault('transform_name', data_config['dataset'])

        return {
            'data': data_config,
            'loader': loader_config,
            'continual': continual_config,
            'transform': transform_config,
            'others': other_config
        }


    def get(self, key, default=None):
        return self.regime.config.get(key, default)