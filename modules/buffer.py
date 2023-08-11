import ctypes
import horovod.torch as hvd
import logging
import math
import random
import torch

from train.train import measure_performance
from utils.meters import get_timer
from utils.log import get_shared_logging_level, display


class AugmentedMinibatch:
    def __init__(self, num_samples, shape, device, total_num_classes):
        self.x = torch.zeros(num_samples, *shape, device=device)
        self.y = torch.randint(high=1000, size=(num_samples,), device=device)
        self.w = torch.ones(num_samples, device=device)
        self.logits = torch.zeros(num_samples, total_num_classes)  # TODO


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

        self.num_samples_per_representative = 1
        self.augmentations_enabled = False
        self.rehearsal_size = 0
        self.rehearsal_tensor = None
        self.rehearsal_metadata = None
        self.rehearsal_counts = None
        self.rehearsal_logits = None

        self.high_performance = True
        try:
            global neomem
            import neomem
        except ImportError:
            logging.info(
                f"Neomem is not installed (high-performance distributed"
                " rehearsal buffer), fallback to a low-performance, local"
                " rehearsal buffer"
            )
            if implementation != "flyweight":
                self.implementation = "flyweight"
                logging.info("Using flyweight buffer implementation")
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
        logging.info(
            f"Initializing the buffer with storage for {self.budget_per_class} representatives per class"
        )
        device = "cuda" if cuda else "cpu"

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
                self.num_samples_per_representative,
                list(self.sample_shape),
                neomem.CPUBuffer,
                discover_endpoints,
                get_shared_logging_level() < logging.INFO,
            )

            self.init_augmented_minibatch(device=device)
            self.dsl.start()
        else:
            size = (
                self.total_num_classes
                * self.budget_per_class
                * self.num_samples_per_representative
            )
            self.rehearsal_tensor = torch.empty(
                [size] + list(self.sample_shape), device=device
            )
            self.rehearsal_metadata = [[0, 1.0] for _ in range(self.total_num_classes)]
            self.rehearsal_counts = [0] * self.total_num_classes

            self.rehearsal_logits = torch.empty(
                [size, self.total_num_classes], device=device
            )

            self.init_augmented_minibatch(device=device)

    def init_augmented_minibatch(self, device="gpu"):
        self.minibatches_ahead = 1 if self.implementation == "flyweight" else 2
        num_samples = (
            self.num_representatives
            if self.implementation == "flyweight"
            else self.num_representatives + self.batch_size
        )

        for i in range(self.minibatches_ahead):
            self.next_minibatches.append(
                AugmentedMinibatch(
                    num_samples, self.sample_shape, device, self.total_num_classes
                )
            )

        if self.high_performance and self.implementation == "flyweight":
            self.dsl.use_these_allocated_variables(
                self.next_minibatches[0].x,
                self.next_minibatches[0].y,
                self.next_minibatches[0].w,
            )

    def update(self, x, y, step, batch_metrics=None):
        """
        Updates the buffer with incoming data. Get some data from (x, y) pairs.
        """
        if self.high_performance:
            self.dsl.measure_performance(measure_performance(step))

        aug_size = self.__get_data(step, batch_metrics=batch_metrics)

        # print("[+] Aug_size : ",aug_size)

        # Assemble the minibatch
        with get_timer(
            "assemble",
            step,
            batch_metrics=batch_metrics,
            dummy=not measure_performance(step),
        ):
            minibatch = self.__get_current_augmented_minibatch(step)
            if self.implementation == "flyweight":
                w = torch.ones(x.size(0), device=x.device)
                concat_x = torch.cat((x, minibatch.x[:aug_size]))
                concat_y = torch.cat((y, minibatch.y[:aug_size]))
                concat_w = torch.cat((w, minibatch.w[:aug_size]))
            else:
                concat_x = minibatch.x[:aug_size]
                concat_y = minibatch.y[:aug_size]
                concat_w = minibatch.w[:aug_size]

        self.add_data(x, y, step, batch_metrics=batch_metrics)

        return concat_x, concat_y, concat_w

    def update_with_logits(self, x, y, logits, step, batch_metrics=None):
        """
        Updates the buffer with incoming data. Get some data from (x, y) pairs.
        """
        if self.high_performance:
            self.dsl.measure_performance(measure_performance(step))

        aug_size = self.__get_data(step, batch_metrics=batch_metrics)

        # print("[+] Aug_size : ",aug_size)

        # Assemble the minibatch
        with get_timer(
            "assemble",
            step,
            batch_metrics=batch_metrics,
            dummy=not measure_performance(step),
        ):
            minibatch = self.__get_current_augmented_minibatch(step)
            if self.implementation == "flyweight":
                w = torch.ones(x.size(0), device=x.device)
                concat_x = torch.cat((x, minibatch.x[:aug_size]))
                concat_y = torch.cat((y, minibatch.y[:aug_size]))
                concat_w = torch.cat((w, minibatch.w[:aug_size]))
            else:
                concat_x = minibatch.x[:aug_size]
                concat_y = minibatch.y[:aug_size]
                concat_w = minibatch.w[:aug_size]

        self.add_data_with_logits(x, y, logits, step, batch_metrics=batch_metrics)

        return concat_x, concat_y, minibatch.logits, concat_w

    def __get_data(self, step, batch_metrics=None):
        aug_size = 0

        if self.high_performance:
            # Get the representatives
            with get_timer(
                "wait",
                step,
                batch_metrics=batch_metrics,
                previous_iteration=True,
                dummy=not measure_performance(step),
            ):
                aug_size = self.dsl.wait()
                n = (
                    aug_size
                    if self.implementation == "flyweight"
                    else aug_size - self.batch_size
                )
                if n > 0:
                    logging.debug(f"Received {n} samples from other nodes")

                if measure_performance(step) and batch_metrics is not None:
                    batch_metrics.add(
                        step,
                        self.dsl.get_metrics(step["batch"]),
                    )
        else:
            if not self.augmentations_enabled:
                return 0

            minibatch = self.__get_current_augmented_minibatch(step)

            choices = random.sample(
                range(self.total_num_classes * self.budget_per_class),
                self.num_representatives,
            )

            classes_to_sample = {}
            for index in choices:
                rehearsal_class_index = math.floor(index / self.budget_per_class)
                num_zeros = sum(
                    1 for metadata in self.rehearsal_metadata if metadata[0] == 0
                )
                rehearsal_class_index %= len(self.rehearsal_metadata) - num_zeros

                # calculate the right value for i
                j = -1
                index_no_empty = 0
                for index_no_empty in range(len(self.rehearsal_metadata)):
                    if self.rehearsal_metadata[index_no_empty][0] == 0:
                        continue
                    j += 1
                    if j == rehearsal_class_index:
                        break
                rehearsal_repr_of_class_index = (
                    index % self.budget_per_class
                ) % self.rehearsal_metadata[index_no_empty][0]

                if index_no_empty not in classes_to_sample.keys():
                    classes_to_sample[index_no_empty] = {
                        "weight": self.rehearsal_metadata[index_no_empty][1],
                        "indices": [],
                    }
                classes_to_sample[index_no_empty]["indices"].append(
                    index_no_empty * self.budget_per_class
                    + rehearsal_repr_of_class_index
                )

            for k, v in classes_to_sample.items():
                for index in v["indices"]:
                    ti = (torch.tensor([aug_size]),)
                    minibatch.x.index_put_(ti, self.rehearsal_tensor[index])
                    minibatch.y.index_put_(ti, torch.tensor(k))
                    minibatch.w.index_put_(ti, torch.tensor(v["weight"]))
                    minibatch.logits.index_put_(ti, self.rehearsal_logits[index])
                    aug_size += 1

        return aug_size

    def add_data(self, x, y, step, batch_metrics=None):
        """
        Fills the rehearsal buffer with (x, y) pairs, sampled randomly from the
        incoming batch of data. Only `num_candidates` will be added to the
        buffer.

        :param x: tensor containing input images
        :param y: tensor containing targets
        :param step: step number, for logging purposes
        """
        if self.high_performance:
            with get_timer(
                "accumulate",
                step,
                batch_metrics=batch_metrics,
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
            for i in range(len(y)):
                label = y[i].item()
                self.rehearsal_metadata[0]
                index = -1
                if self.rehearsal_metadata[label][0] < self.budget_per_class:
                    index = self.rehearsal_metadata[label][0]
                else:
                    if random.randint(0, self.batch_size - 1) >= self.num_candidates:
                        continue
                    index = random.randint(0, self.budget_per_class - 1)

                for r in range(0, self.num_samples_per_representative):
                    j = self.budget_per_class * label + index + r
                    self.rehearsal_tensor.index_put_((torch.tensor([j]),), x[i])

                if index >= self.rehearsal_metadata[label][0]:
                    self.rehearsal_size += 1
                    self.rehearsal_metadata[label][0] += 1

                self.rehearsal_counts[label] += 1

    def add_data_with_logits(self, x, y, logits, step, batch_metrics=None):
        """
        Fills the rehearsal buffer with (x, y) pairs, sampled randomly from the
        incoming batch of data. Only `num_candidates` will be added to the
        buffer.

        :param x: tensor containing input images
        :param y: tensor containing targets
        :param step: step number, for logging purposes
        """
        if self.high_performance:
            with get_timer(
                "accumulate",
                step,
                batch_metrics=batch_metrics,
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
            for i in range(len(y)):
                label = y[i].item()
                self.rehearsal_metadata[0]
                index = -1
                if self.rehearsal_metadata[label][0] < self.budget_per_class:
                    index = self.rehearsal_metadata[label][0]
                else:
                    if random.randint(0, self.batch_size - 1) >= self.num_candidates:
                        continue
                    index = random.randint(0, self.budget_per_class - 1)

                for r in range(0, self.num_samples_per_representative):
                    j = self.budget_per_class * label + index + r
                    self.rehearsal_tensor.index_put_((torch.tensor([j]),), x[i])
                    self.rehearsal_logits.index_put_((torch.tensor([j]),), logits[i])

                if index >= self.rehearsal_metadata[label][0]:
                    self.rehearsal_size += 1
                    self.rehearsal_metadata[label][0] += 1

                self.rehearsal_counts[label] += 1

    def enable_augmentations(self, enabled=True):
        self.augmentations_enabled = enabled
        if self.high_performance:
            self.dsl.enable_augmentation(enabled)

    def __get_current_augmented_minibatch(self, step):
        return self.next_minibatches[step["batch"] % self.minibatches_ahead]

    def __get_next_augmented_minibatch(self, step):
        return self.next_minibatches[(step["batch"] + 1) % self.minibatches_ahead]

    def __len__(self):
        return self.get_size()

    def get_size(self):
        if self.high_performance:
            return self.dsl.get_rehearsal_size()
        return self.rehearsal_size
