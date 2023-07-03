import abc
import copy
import ctypes
import horovod.torch as hvd
import numpy as np
import logging
import torch

from torch import nn
from torch.nn import functional as F

from train.train import measure_performance
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

    def __init__(
        self,
        total_num_classes,
        sample_shape,
        batch_size,
        budget_per_class=16,
        num_representatives=8,
        num_candidates=8,
        provider="na+sm",
        discover_endpoints=True,
        cuda=True,
        cuda_rdma=False,
        mode="separate",
        implementation="standard",
    ):
        assert mode in ("separate")
        assert implementation in ("standard", "flyweight")

        self.total_num_classes = total_num_classes
        self.sample_shape = sample_shape
        self.batch_size = batch_size
        self.budget_per_class = budget_per_class
        self.num_representatives = num_representatives
        self.num_candidates = num_candidates
        self.next_minibatches = []
        self.mode = mode
        self.implementation = implementation

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

        self.init_buffer(
            provider=provider,
            cuda=cuda,
            cuda_rdma=cuda_rdma,
            discover_endpoints=discover_endpoints,
        )

    def init_buffer(
        self, provider=None, cuda=True, cuda_rdma=False, discover_endpoints=True
    ):
        """
        Initializes the memory required to store representatives and their
        labels. If `high_performance` is enabled, this instanciates Neomem.
        """
        device = "cuda" if cuda else "cpu"
        num_samples_per_representative = 1

        if self.high_performance:
            engine = neomem.EngineLoader(
                provider, ctypes.c_uint16(hvd.rank()).value, cuda_rdma
            )
            self.dsl = neomem.DistributedStreamLoader(
                engine,
                neomem.Classification,
                self.total_num_classes,
                self.budget_per_class,
                self.num_representatives,
                self.num_candidates,
                ctypes.c_int64(torch.random.initial_seed() + hvd.rank()).value,
                num_samples_per_representative,
                list(self.sample_shape),
                neomem.CPUBuffer,
                discover_endpoints,
                get_logging_level() not in ("info"),
            )

            self.init_augmented_minibatch(device=device)
            self.dsl.start()
        else:
            size = total_num_classes * budget_per_class * num_samples_per_representative
            storage = tensor.empty([size] + sample_shape, device=device)

    def init_augmented_minibatch(self, device="gpu"):
        self.minibatches_ahead = 1 if self.implementation == "flyweight" else 2
        num_samples = (
            self.num_representatives
            if self.implementation == "flyweight"
            else self.num_representatives + self.batch_size
        )

        for i in range(self.minibatches_ahead):
            self.next_minibatches.append(
                AugmentedMinibatch(num_samples, self.sample_shape, device)
            )

        if self.implementation == "flyweight":
            self.dsl.use_these_allocated_variables(
                self.next_minibatches[0].x,
                self.next_minibatches[0].y,
                self.next_minibatches[0].w,
            )

    def update(self, x, y, w, step, perf_metrics=None):
        """
        Updates the buffer with incoming data. Get some data from (x, y) pairs.
        """
        flyweight = self.implementation == "flyweight"
        if not self.init:
            self.add_data(x, y, step)
            self.init = True

        # Get the representatives
        with get_timer(
            "wait",
            step["batch"],
            perf_metrics=perf_metrics,
            previous_iteration=True,
            dummy=not measure_performance(step),
        ):
            aug_size = self.dsl.wait()
            n = aug_size if flyweight else aug_size - self.batch_size
            if n > 0:
                logging.debug(f"Received {n} samples from other nodes")

            if measure_performance and perf_metrics is not None:
                cpp_metrics = self.dsl.get_metrics(step["batch"])
                perf_metrics.add(step["batch"] - 1, cpp_metrics)

        # Assemble the minibatch
        with get_timer(
            "assemble",
            step["batch"],
            perf_metrics=perf_metrics,
            dummy=not measure_performance(step),
        ):
            minibatch = self.__get_current_augmented_minibatch(step)
            if flyweight:
                concat_x = torch.cat((x, minibatch.x[:aug_size]))
                concat_y = torch.cat((y, minibatch.y[:aug_size]))
                concat_w = torch.cat((w, minibatch.w[:aug_size]))
            else:
                concat_x = minibatch.x[:aug_size]
                concat_y = minibatch.y[:aug_size]
                concat_w = minibatch.w[:aug_size]

        self.add_data(x, y, step, perf_metrics=perf_metrics)

        return concat_x, concat_y, concat_w

    def add_data(self, x, y, step, perf_metrics=None):
        """
        Fills the rehearsal buffer with (x, y) pairs.

        :param x: tensor containing input images
        :param y: tensor containing targets
        :param step: step number, for logging purposes
        """
        if self.high_performance:
            with get_timer(
                "accumulate",
                step["batch"],
                perf_metrics=perf_metrics,
                dummy=not measure_performance(step),
            ):
                if self.implementation == "flyweight":
                    self.dsl.accumulate(x, y)
                else:
                    next_minibatch = self.__get_next_augmented_minibatch(step)
                    self.dsl.accumulate(
                        x, y, next_minibatch.x, next_minibatch.y, next_minibatch.w
                    )
        else:
            pass
            """
            for i in range(self.batch_size):
                label = 0
                if (task_type == neomem.Classification)
                    label = y[i].item()

                index = -1
                if rehearsal_metadata[label].first < N:
                    index = rehearsal_metadata[label].first
                else:
                    if (dice_candidate(rand_gen) >= C):
                        continue
                    index = dice_buffer(rand_gen)

                for (size_t r = 0; r < num_samples_per_representative; r++):
                    size_t j = N * label + index + r
                    rehearsal_tensor->index_put_({static_cast<int>(j)}, batch.samples.index({i}))

                if index >= rehearsal_metadata[label].first:
                    rehearsal_size++
                    rehearsal_metadata[label].first++

                rehearsal_counts[label]++
            """

    def enable_augmentations(self, enabled=True):
        self.dsl.enable_augmentation(enabled)

    def __get_current_augmented_minibatch(self, step):
        return self.next_minibatches[step["batch"] % self.minibatches_ahead]

    def __get_next_augmented_minibatch(self, step):
        return self.next_minibatches[(step["batch"] + 1) % self.minibatches_ahead]

    def __len__(self):
        return self.get_size()

    def get_size(self):
        return self.dsl.get_rehearsal_size()
