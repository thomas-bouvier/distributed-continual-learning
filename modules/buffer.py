import abc
import copy
import ctypes
import horovod.torch as hvd
import numpy as np
import logging
import torch

from torch import nn
from torch.nn import functional as F

from utils.meters import get_timer
from utils.log import get_logging_level


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
        mode='standard'):
        assert mode in ('standard', 'flyweight')

        self.sample_shape = sample_shape
        self.batch_size = batch_size
        self.budget_per_class = budget_per_class
        self.num_candidates = num_candidates
        self.num_representatives = num_representatives
        self.next_minibatches = []
        self.mode = mode
        self.init = False
        self.high_performance = True

        try:
            global neomem
            import neomem
        except ImportError:
            raise ImportError(
                f"Neomem is not installed (high-performance distributed"
                " rehearsal buffer), fallback to a low-performance, local"
                " rehearsal buffer."
            )
            self.high_performance = False

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
        self.minibatches_ahead = 1 if self.mode == 'flyweight' else 2
        num_samples = self.num_representatives if self.mode == 'flyweight' else self.num_representatives + self.batch_size

        for i in range(self.minibatches_ahead):
            self.next_minibatches.append(
                AugmentedMinibatch(num_samples, self.sample_shape, 'cuda' if cuda else 'cpu')
            )

        if self.mode == 'flyweight':
            self.dsl.use_these_allocated_variables(self.next_minibatches[0].x,
                                                   self.next_minibatches[0].y,
                                                   self.next_minibatches[0].w)


    def update(self, x, y, w, step, measure_performance=False):
        """Get some data from (x, y) pairs."""
        flyweight = self.mode == 'flyweight'
        if not self.init:
            self.add_data(x, y, step)
            self.init = True

        # Get the representatives
        with get_timer('wait', step["batch"], previous_iteration=True):
            aug_size = self.dsl.wait()
            n = aug_size if flyweight else aug_size - self.batch_size
            if n > 0:
                logging.debug(f"Received {n} samples from other nodes")

            if measure_performance:
                cpp_metrics = self.dsl.get_metrics(step["batch"])
                self.perf_metrics.add(step["batch"] - 1, cpp_metrics)

        # Assemble the minibatch
        with get_timer('assemble', step["batch"]):
            minibatch = self.__get_current_augmented_minibatch(step)
            if flyweight:
                concat_x = torch.cat((x, minibatch.x[:aug_size]))
                concat_y = torch.cat((y, minibatch.y[:aug_size]))
                concat_w = torch.cat((w, minibatch.w[:aug_size]))
            else:
                concat_x = minibatch.x[:aug_size]
                concat_y = minibatch.y[:aug_size]
                concat_w = minibatch.w[:aug_size]

        self.add_data(x, y, step)

        return concat_x, concat_y, concat_w


    def add_data(self, x, y, step):
        """Fill the rehearsal buffer with (x, y) pairs."""
        with get_timer('accumulate', step["batch"]):
            if self.mode == 'flyweight':
                self.dsl.accumulate(x, y)
            else:
                next_minibatch = self.__get_next_augmented_minibatch(step)
                self.dsl.accumulate(
                    x, y, next_minibatch.x, next_minibatch.y, next_minibatch.w
                )


    def enable_augmentations(self, enabled=True):
        self.dsl.enable_augmentation(enabled)


    def __get_current_augmented_minibatch(self, step):
        return self.next_minibatches[step["batch"] % self.minibatches_ahead]


    def __get_next_augmented_minibatch(self, step):
        return self.next_minibatches[(step["batch"] + 1) % self.minibatches_ahead]


    def get_size(self):
        return self.dsl.get_rehearsal_size()
