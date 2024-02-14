import ctypes
import horovod.torch as hvd
import logging
import math
import random
import torch

from train.train import measure_performance
from utils.meters import get_timer
from utils.log import get_shared_logging_level


class AugmentedMinibatch:
    def __init__(
        self,
        num_samples,
        shape,
        device,
        num_samples_per_representative=1,
        num_samples_per_activation=0,
    ):
        self.input = torch.zeros(num_samples, *shape, device=device)
        self.ground_truth = [
            torch.zeros(num_samples, *shape, device=device)
            for _ in range(num_samples_per_representative - 1)
        ]
        self.labels = torch.randint(high=1000, size=(num_samples,), device=device)
        self.weights = torch.ones(num_samples, device=device)

        # todo: we should have a way to support different shapes
        self.logits = [
            torch.zeros(num_samples, *shape, device=device, dtype=torch.float16)
            for _ in range(num_samples_per_activation)
        ]


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
        augmentations_offset=0,
        soft_augmentations_offset=0,
        provider="na+sm",
        discover_endpoints=True,
        cuda=True,
        cuda_rdma=False,
        mode="separate",
        implementation="standard",
        num_samples_per_representative=1,
        num_samples_per_activation=0,
    ):
        assert mode in ("separate")
        assert implementation in ("standard", "flyweight")

        self.total_num_classes = total_num_classes
        self.sample_shape = sample_shape
        self.batch_size = batch_size
        self.budget_per_class = budget_per_class
        self.num_representatives = num_representatives
        self.num_candidates = num_candidates
        self.augmentations_offset = augmentations_offset
        self.soft_augmentations_offset = soft_augmentations_offset
        self.next_minibatches = []
        self.next_minibatches_derpp = []
        self.mode = mode
        self.implementation = implementation

        self.num_samples_per_representative = num_samples_per_representative
        self.num_samples_per_activation = num_samples_per_activation
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
                "Neomem is not installed (high-performance distributed"
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
            "Initializing the buffer with storage for %d representatives per class",
            self.budget_per_class,
        )
        device = "cuda" if cuda else "cpu"

        if self.high_performance:
            engine = neomem.EngineLoader(
                provider, ctypes.c_uint16(hvd.rank()).value, cuda_rdma
            )
            self.dsl = neomem.DistributedStreamLoader.create(
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
            rehearsal_tensor_size = (
                self.total_num_classes
                * self.budget_per_class
                * self.num_samples_per_representative
            )
            self.rehearsal_tensor = torch.empty(
                [rehearsal_tensor_size] + list(self.sample_shape)
            )
            self.rehearsal_metadata = [[0, 1.0] for _ in range(self.total_num_classes)]
            self.rehearsal_counts = [0] * self.total_num_classes

            # todo: we should have a way to support different shapes
            reharsal_logits_size = (
                self.total_num_classes
                * self.budget_per_class
                * self.num_samples_per_activation
            )
            self.rehearsal_logits = torch.empty(
                [reharsal_logits_size] + list(self.sample_shape), dtype=torch.float16
            )

            self.init_augmented_minibatch(device=device)

    def init_augmented_minibatch(self, device="gpu"):
        self.minibatches_ahead = 1 if self.implementation == "flyweight" else 2
        num_samples = (
            self.num_representatives
            if self.implementation == "flyweight"
            else self.num_representatives + self.batch_size
        )

        for _ in range(self.minibatches_ahead):
            self.next_minibatches.append(
                AugmentedMinibatch(
                    num_samples,
                    self.sample_shape,
                    device,
                    num_samples_per_representative=self.num_samples_per_representative,
                    num_samples_per_activation=self.num_samples_per_activation,
                )
            )
            self.next_minibatches_derpp.append(
                AugmentedMinibatch(
                    num_samples,
                    self.sample_shape,
                    device,
                    num_samples_per_representative=self.num_samples_per_representative,
                    num_samples_per_activation=self.num_samples_per_activation,
                )
            )

        if self.high_performance and self.implementation == "flyweight":
            self.dsl.use_these_allocated_variables(
                self.next_minibatches[0].input,
                self.next_minibatches[0].ground_truth,
                self.next_minibatches[0].labels,
                self.next_minibatches[0].weights,
            )

    def update(
        self, x, y, step, batch_metrics=None, logits=[], ground_truth=[], derpp=False
    ):
        """
        Updates the buffer with incoming data. Get some data from (x, y) pairs.
        """
        if self.num_samples_per_representative > 1:
            assert len(ground_truth) > 0, "Some ground-truth tensors should be provided"

        if self.high_performance:
            self.dsl.measure_performance(measure_performance(step))

        aug_size = self.__get_data(step, batch_metrics=batch_metrics, derpp=derpp)

        # Assemble the minibatch
        with get_timer(
            "assemble",
            step,
            batch_metrics=batch_metrics,
            dummy=not measure_performance(step),
        ):
            minibatch = self.__get_current_augmented_minibatch(step, derpp=derpp)
            concat_ground_truth = []
            concat_logits = []

            if self.implementation == "flyweight":
                w = torch.ones(x.size(0), device=x.device)
                concat_x = torch.cat((x, minibatch.input[:aug_size]))
                for i, v in enumerate(ground_truth):
                    concat_ground_truth.append(
                        torch.cat((v, minibatch.ground_truth[i][:aug_size]))
                    )
                concat_y = torch.cat((y, minibatch.labels[:aug_size]))
                concat_w = torch.cat((w, minibatch.weights[:aug_size]))
                for i, v in enumerate(logits):
                    concat_logits.append(torch.cat((v, minibatch.logits[i][:aug_size])))
            else:
                concat_x = minibatch.input[:aug_size]
                for i in range(len(ground_truth)):
                    concat_ground_truth.append(minibatch.ground_truth[i][:aug_size])
                concat_y = minibatch.labels[:aug_size]
                concat_w = minibatch.weights[:aug_size]
                for l in range(len(logits)):
                    concat_logits.append(minibatch.logits[l][:aug_size])

        if not derpp:
            self.add_data(
                x,
                y,
                step,
                batch_metrics=batch_metrics,
                logits=logits,
                ground_truth=ground_truth,
            )

        return concat_x, concat_ground_truth, concat_y, concat_w, concat_logits

    def __get_data(self, step, batch_metrics=None, derpp=False):
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
                    logging.debug("Received %s samples from other nodes", n)

                if measure_performance(step) and batch_metrics is not None:
                    batch_metrics.add(
                        step,
                        self.dsl.get_metrics(step["batch"]),
                    )
        else:
            if not self.augmentations_enabled:
                return 0

            minibatch = self.__get_current_augmented_minibatch(step, derpp=derpp)

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
                for index_no_empty, _ in enumerate(self.rehearsal_metadata):
                    if self.rehearsal_metadata[index_no_empty][0] == 0:
                        continue
                    j += 1
                    if j == rehearsal_class_index:
                        break
                rehearsal_repr_of_class_index = (
                    index % self.budget_per_class
                ) % self.rehearsal_metadata[index_no_empty][0]

                if index_no_empty not in classes_to_sample:
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
                    device = minibatch.input.device

                    rehearsal_tensor_index = index * self.num_samples_per_representative

                    # input tensor
                    minibatch.input.index_put_(
                        ti, self.rehearsal_tensor[rehearsal_tensor_index].to(device)
                    )

                    # other ground_truth tensors
                    for r in range(0, self.num_samples_per_representative - 1):
                        k = rehearsal_tensor_index + r + 1
                        minibatch.ground_truth[r].index_put_(
                            ti, self.rehearsal_tensor[k].to(device)
                        )

                    # logits
                    rehearsal_logits_index = index * self.num_samples_per_activation
                    for l in range(0, self.num_samples_per_activation):
                        m = rehearsal_logits_index + l
                        minibatch.logits[l].index_put_(
                            ti, self.rehearsal_logits[m].to(device)
                        )

                    minibatch.labels.index_put_(ti, torch.tensor(k, device=device))
                    minibatch.weights.index_put_(
                        ti, torch.tensor(v["weight"], device=device)
                    )

                    aug_size += 1

        return aug_size

    def add_data(self, x, y, step, batch_metrics=None, logits=[], ground_truth=[]):
        """
        Fills the rehearsal buffer with (x, y) pairs, sampled randomly from the
        incoming batch of data. Only `num_candidates` will be added to the
        buffer.

        :param x: tensor containing input images
        :param y: tensor containing targets
        :param step: step number, for logging purposes
        """
        if self.num_samples_per_representative > 1:
            assert (
                len(ground_truth) == self.num_samples_per_representative - 1
            ), "Some ground-truth tensors should be provided"

        if self.num_samples_per_activation > 0:
            assert (
                len(logits) == self.num_samples_per_activation
            ), "Some logits should be provided"

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
                        x,
                        y,
                        ground_truth,
                        next_minibatch.input,
                        next_minibatch.labels,
                        next_minibatch.weights,
                        next_minibatch.ground_truth,
                    )
        else:
            for i in range(y.shape[0]):
                label = y[i].item()

                # picking an index to replace/append to
                index = -1
                if self.rehearsal_metadata[label][0] < self.budget_per_class:
                    index = self.rehearsal_metadata[label][0]
                else:
                    if random.randint(0, self.batch_size - 1) >= self.num_candidates:
                        continue
                    index = random.randint(0, self.budget_per_class - 1)

                # input tensor
                rehearsal_tensor_index = (
                    self.budget_per_class * label + index
                ) * self.num_samples_per_representative
                self.rehearsal_tensor.index_put_(
                    (torch.tensor([rehearsal_tensor_index]),), x[i].cpu()
                )

                # other ground_truth tensors
                for r in range(0, self.num_samples_per_representative - 1):
                    k = rehearsal_tensor_index + r + 1
                    self.rehearsal_tensor.index_put_(
                        (torch.tensor([k]),), ground_truth[r][i].cpu()
                    )

                # logits
                rehearsal_logits_index = (
                    self.budget_per_class * label + index
                ) * self.num_samples_per_activation
                for l in range(0, self.num_samples_per_activation):
                    m = rehearsal_logits_index + l
                    self.rehearsal_logits.index_put_(
                        (torch.tensor([m]),), logits[l][i].cpu()
                    )

                if index >= self.rehearsal_metadata[label][0]:
                    self.rehearsal_size += 1
                    self.rehearsal_metadata[label][0] += 1

                self.rehearsal_counts[label] += 1

    def enable_augmentations(self, enabled=True):
        self.augmentations_enabled = enabled
        if self.high_performance:
            self.dsl.enable_augmentation(enabled)

    def __get_current_augmented_minibatch(self, step, derpp=False):
        if derpp:
            return self.next_minibatches_derpp[step["batch"] % self.minibatches_ahead]
        return self.next_minibatches[step["batch"] % self.minibatches_ahead]

    def __get_next_augmented_minibatch(self, step, derpp=False):
        if derpp:
            return self.next_minibatches_derpp[
                (step["batch"] + 1) % self.minibatches_ahead
            ]
        return self.next_minibatches[(step["batch"] + 1) % self.minibatches_ahead]

    def __len__(self):
        return self.get_size()

    def get_size(self):
        if self.high_performance:
            return self.dsl.get_rehearsal_size()
        return self.rehearsal_size
