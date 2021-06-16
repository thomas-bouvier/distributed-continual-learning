from enum import Enum

class Dataset(Enum):
    MNIST = 1
    CIFAR10 = 2
    CIFAR100 = 3

class Model(Enum):
    NET = 1