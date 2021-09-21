import os
import torch
import horovod.torch as hvd
from filelock import FileLock

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets
from copy import deepcopy

from preprocess import get_transform
from regime import Regime

def get_dataset(dataset='mnist', split='train', transform=None,
    target_transform=None, download=False, dataset_dir='./data'):
    train = (split == 'train')
    root = os.path.expanduser(dataset_dir)

    if dataset == 'mnist':
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.MNIST(root=root,
                            train=train,
                            transform=transform,
                            target_transform=target_transform,
                            download=download)
    elif dataset == 'cifar10':
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.CIFAR10(root=os.path.join(root, 'CIFAR10'),
                                    train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
    elif dataset == 'cifar100':
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            return datasets.CIFAR100(root=os.path.join(root, 'CIFAR100'),
                                    train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
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
                    'dataset_dir'}
_DATALOADER_ARGS = {'batch_size', 'shuffle', 'sampler', 'batch_sampler',
                    'num_workers', 'collate_fn', 'pin_memory', 'drop_last',
                    'timeout', 'worker_init_fn'}
_TRANSFORM_ARGS = {'transform_name'}
_OTHER_ARGS = {'distributed'}


"""
Inspired by https://github.com/eladhoffer/convNet.pytorch/blob/master/data.py
"""
class DataRegime(object):
    def __init__(self, regime, hvd, defaults={}):
        self.regime = Regime(regime, deepcopy(defaults))
        self.hvd = hvd
        self.epoch = 0
        self.steps = None
        self.get_loader(True)


    def get_loader(self, force_update=False):
        if force_update:
            config = self.get_config()

            self._transform = get_transform(**config['transform'])
            config['data'].setdefault('transform', self._transform)
            self._data = get_dataset(**config['data'])

            if config['others'].get('distributed', False):
                config['loader']['sampler'] = DistributedSampler(
                    self._data, num_replicas=hvd.size(), rank=hvd.rank()
                )
                config['loader']['shuffle'] = None
                # pin-memory currently broken for distributed
                config['loader']['pin_memory'] = False

            self._sampler = config['loader'].get('sampler', None)
            self._loader = DataLoader(self._data, **config['loader'])

        return self._loader


    def set_epoch(self, epoch):
        self.epoch = epoch
        if self._sampler is not None and hasattr(self._sampler, 'set_epoch'):
            self._sampler.set_epoch(epoch)


    def __len__(self):
        return len(self._data)


    def __repr__(self):
        return str(self.regime)


    def get_config(self):
        config = self.regime.config
        loader_config = {
            k: v for k, v in config.items() if k in _DATALOADER_ARGS}
        data_config = {
            k: v for k, v in config.items() if k in _DATA_ARGS}
        transform_config = {
            k: v for k, v in config.items() if k in _TRANSFORM_ARGS}
        other_config = {
            k: v for k, v in config.items() if k in _OTHER_ARGS}

        transform_config.setdefault('transform_name', data_config['dataset'])

        return {
            'data': data_config,
            'loader': loader_config,
            'transform': transform_config,
            'others': other_config
        }


    def get(self, key, default=None):
        return self.regime.config.get(key, default)