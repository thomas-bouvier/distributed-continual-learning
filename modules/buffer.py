import abc
import copy
import ctypes
import horovod.torch as hvd
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from utils.log import get_logging_level

import neomem


class AugmentedMinibatch:
    def __init__(self, num_samples, shape, device):
        self.x = torch.zeros(num_samples, *shape, device=device)
        self.y = torch.randint(high=1000, size=(num_samples,), device=device)
        self.w = torch.ones(num_samples, device=device)


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, total_num_classes, sample_shape, batch_size,
        budget_per_class=16, num_candidates=8, num_representatives=8,
        provider='na+sm', discover_endpoints=True, cuda=True, cuda_rdma=False,
        mode='flyweight'):
        assert mode in ('standard', 'flyweight')

        self.sample_shape = sample_shape
        self.budget_per_class = budget_per_class
        self.num_candidates = num_candidates
        self.num_representatives = num_representatives
        self.next_minibatches = []
        self.mode = mode

        engine = neomem.EngineLoader(provider,
            ctypes.c_uint16(hvd.rank()).value, cuda_rdma
        )
        self.dsl = neomem.DistributedStreamLoader(
            engine, neomem.Classification,
            total_num_classes, budget_per_class, num_representatives, num_candidates,
            ctypes.c_int64(torch.random.initial_seed() + hvd.rank()).value,
            1, list(sample_shape), neomem.CPUBuffer, discover_endpoints, get_logging_level() not in ('info')
        )

        self.init_augmented_minibatch(cuda=cuda)
        self.dsl.start()


    def init_augmented_minibatch(self, cuda=True):
        minibatches_ahead = 1 if self.mode == 'flyweight' else 2
        num_samples = self.num_representatives if self.mode == 'flyweight' else self.num_representatives + self.batch_size

        for i in range(minibatches_ahead):
            self.next_minibatches.append(
                AugmentedMinibatch(num_samples, self.sample_shape, 'cuda' if cuda else 'cpu')
            )

        if self.mode == 'flyweight':
            self.dsl.use_these_allocated_variables(self.next_minibatches[0].x,
                                                   self.next_minibatches[0].y,
                                                   self.next_minibatches[0].w)


    def add_data(self, x, y):
        if self.mode == 'flyweight':
            self.dsl.accumulate(x, y)
        else:
            self.dsl.accumulate(x, y, self.next_minibatches[0].x,
                                      self.next_minibatches[0].y,
                                      self.next_minibatches[0].w)


    def get_data(self):
        self.dsl.enable_augmentation(True)
        return __get_current_augmented_minibatch()


    def __get_current_augmented_minibatch(self):
        return self.next_minibatches[self.global_batch % self.minibatches_ahead]


    def __get_next_augmented_minibatch(self):
        return self.next_minibatches[(self.global_batch + 1) % self.minibatches_ahead]