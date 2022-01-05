import os

from continuum import datasets as datasets_c
from filelock import FileLock
from torchvision import datasets

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
