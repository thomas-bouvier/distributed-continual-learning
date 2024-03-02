import ctypes
import horovod.torch as hvd
import logging
import math
import random
import torch

from typing import List

from train.train import measure_performance
from utils.meters import get_timer
from utils.log import get_shared_logging_level


class AugmentedMinibatch:
    def __init__(
        self,
        num_samples,
        representative_shape,
        device,
        num_samples_per_representative=1,
        num_samples_distillation=0,
        activation_shape=None,
        num_samples_per_activation=0,
    ):
        self.device = device
        self.samples = [
            torch.zeros(num_samples, *representative_shape, device=device)
            for _ in range(num_samples_per_representative)
        ]
        self.labels = torch.randint(high=1000, size=(num_samples,), device=device)
        self.weights = torch.ones(num_samples, device=device)

        self.activations = [
            torch.zeros(num_samples_distillation, *activation_shape, device=device)
            for _ in range(num_samples_per_activation)
        ]
        self.activations_rep = torch.zeros(
            num_samples_distillation, *representative_shape, device=device
        )


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(
        self,
        total_num_classes,
        representative_shape,
        batch_size,
        budget_per_class=16,
        num_representatives=8,
        num_representatives_distillation=8,
        num_candidates=8,
        augmentations_offset=0,
        soft_augmentations_offset=0,
        activation_shape=None,
        provider="na+sm",
        discover_endpoints=True,
        cuda=True,
        cuda_rdma=False,
        implementation="standard",
        num_samples_per_representative=1,
        num_samples_per_activation=0,
    ):
        assert implementation in ("standard", "flyweight")

        self.total_num_classes = total_num_classes
        self.representative_shape = representative_shape
        self.activation_shape = activation_shape
        self.batch_size = batch_size
        self.budget_per_class = budget_per_class
        self.num_representatives = num_representatives
        self.num_representatives_distillation = num_representatives_distillation
        self.num_candidates = num_candidates
        self.augmentations_offset = augmentations_offset
        self.soft_augmentations_offset = soft_augmentations_offset
        self.next_minibatches = []
        self.implementation = implementation
        self.minibatches_ahead = 1 if self.implementation == "flyweight" else 2

        self.num_samples_per_representative = num_samples_per_representative
        self.num_samples_per_activation = num_samples_per_activation
        self.augmentations_enabled = False
        self.rehearsal_size = 0
        self.rehearsal_tensor = None
        self.rehearsal_metadata = None
        self.rehearsal_counts = None
        self.rehearsal_activations = None

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
            # The engine should be stored.
            self.engine = neomem.EngineLoader(
                provider, ctypes.c_uint16(hvd.rank()).value, cuda_rdma
            )
            self.dsl = neomem.DistributedStreamLoader.create(
                self.engine,
                neomem.Rehearsal,
                self.total_num_classes,
                self.budget_per_class,
                self.num_candidates,
                ctypes.c_int64(torch.random.initial_seed() + hvd.rank()).value,
                self.num_representatives,
                self.num_samples_per_representative,
                self.representative_shape,
                self.num_representatives_distillation,
                self.num_samples_per_activation,
                self.activation_shape or [],
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
                [rehearsal_tensor_size] + list(self.representative_shape)
            )
            self.rehearsal_metadata = [[0, 1.0] for _ in range(self.total_num_classes)]
            self.rehearsal_counts = [0] * self.total_num_classes

            if self.num_samples_per_activation:
                reharsal_logits_size = (
                    self.total_num_classes
                    * self.budget_per_class
                    * self.num_samples_per_activation
                )
                self.rehearsal_activations = torch.empty(
                    [reharsal_logits_size] + list(self.activation_shape)
                )

            self.init_augmented_minibatch(device=device)

    def init_augmented_minibatch(self, device="gpu"):
        num_samples = (
            self.num_representatives
            if self.implementation == "flyweight"
            else self.num_representatives + self.batch_size
        )

        for _ in range(self.minibatches_ahead):
            self.next_minibatches.append(
                AugmentedMinibatch(
                    num_samples,
                    self.representative_shape,
                    device,
                    num_samples_per_representative=self.num_samples_per_representative,
                    num_samples_per_activation=self.num_samples_per_activation,
                    num_samples_distillation=self.num_representatives_distillation,
                    activation_shape=self.activation_shape,
                )
            )

        if self.high_performance and self.implementation == "flyweight":
            self.dsl.use_these_allocated_variables(
                self.next_minibatches[0].input,
                self.next_minibatches[0].labels,
                self.next_minibatches[0].weights,
                self.next_minibatches[0].activations,
                self.next_minibatches[0].activations_rep,
            )

    def update(
        self,
        samples: List[torch.Tensor],
        targets: torch.Tensor,
        step,
        batch_metrics=None,
        activations=None,
    ):
        """
        Updates the buffer with incoming data. Get some data from (x, y) pairs.
        """
        if activations is None:
            activations = []

        assert len(samples)
        assert len(samples) == self.num_samples_per_representative
        if self.num_samples_per_activation > 0:
            assert (
                len(activations) == self.num_samples_per_activation
            ), "Some activations should be provided"

        if self.high_performance:
            self.dsl.measure_performance(measure_performance(step))

        aug_size1, aug_size2 = self.__get_data(step, batch_metrics=batch_metrics)

        # Assemble the minibatch, consuming the augmented part from the staged
        # buffer.
        with get_timer(
            "assemble",
            step,
            batch_metrics=batch_metrics,
            dummy=not measure_performance(step),
        ):
            minibatch = self.__get_current_augmented_minibatch(step)
            concat_x = []
            concat_activations = []
            concat_activations_rep = []

            if self.implementation == "flyweight":
                w = torch.ones(samples[0].size(0), device=samples[0].device)
                for i, v in enumerate(samples):
                    concat_x.append(torch.cat((v, minibatch.samples[i][:aug_size1])))
                concat_y = torch.cat((targets, minibatch.labels[:aug_size1]))
                concat_w = torch.cat((w, minibatch.weights[:aug_size1]))
            else:
                for i, _ in enumerate(samples):
                    concat_x.append(minibatch.samples[i][:aug_size1])
                concat_y = minibatch.labels[:aug_size1]
                concat_w = minibatch.weights[:aug_size1]

            # Activations and associated representatives are always in "standard"
            # mode.
            for i in range(self.num_samples_per_activation):
                concat_activations.append(minibatch.activations[i][:aug_size2])
            if activations:
                concat_activations_rep = minibatch.activations_rep[:aug_size2]

        self.add_data(
            samples,
            targets,
            step,
            batch_metrics=batch_metrics,
            activations=activations,
        )

        return concat_x, concat_y, concat_w, concat_activations, concat_activations_rep

    def __get_random_index(self, nsamples):
        """Returns a dict containing classes as keys and weights/corresponding indices
        as values.

        nsamples: number of indices to sample
        """
        choices = random.sample(
            range(self.total_num_classes * self.budget_per_class),
            nsamples,
        )

        classes_to_sample = {}
        for index in choices:
            rehearsal_class_index = math.floor(index / self.budget_per_class)
            num_zeros = sum(
                1 for metadata in self.rehearsal_metadata if metadata[0] == 0
            )

            if len(self.rehearsal_metadata) - num_zeros == 0:
                continue
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
                index_no_empty * self.budget_per_class + rehearsal_repr_of_class_index
            )

        return classes_to_sample

    def __get_data(self, step, batch_metrics=None):
        aug_size1 = 0
        aug_size2 = 0

        if self.high_performance:
            # Get the representatives
            with get_timer(
                "wait",
                step,
                batch_metrics=batch_metrics,
                previous_iteration=True,
                dummy=not measure_performance(step),
            ):
                aug_size1, aug_size2 = self.dsl.wait()
                n = (
                    aug_size1
                    if self.implementation == "flyweight"
                    else aug_size1 - self.batch_size
                )
                if n > 0:
                    logging.debug("Received %s representatives from other nodes", n)
                if aug_size2 > 0:
                    logging.debug("Received %s activations from other nodes", aug_size2)

                if measure_performance(step) and batch_metrics is not None:
                    batch_metrics.add(
                        step,
                        self.dsl.get_metrics(step["batch"]),
                    )
        else:
            if not self.augmentations_enabled:
                return 0, 0

            if self.num_samples_per_representative:
                aug_size1 = self.__get_representatives_data(step)
            if self.num_samples_per_activation:
                aug_size2 = self.__get_activations_data(step)

        return aug_size1, aug_size2

    def __get_representatives_data(self, step):
        aug_size = 0

        minibatch = self.__get_current_augmented_minibatch(step)
        classes_to_sample = self.__get_random_index(self.num_representatives)

        for classn, v in classes_to_sample.items():
            for index in v["indices"]:
                ti = (torch.tensor([aug_size]),)
                device = minibatch.device

                # representatives
                rehearsal_tensor_index = index * self.num_samples_per_representative
                for r in range(0, self.num_samples_per_representative):
                    k = rehearsal_tensor_index + r
                    minibatch.samples[r].index_put_(
                        ti, self.rehearsal_tensor[k].to(device)
                    )

                    minibatch.labels.index_put_(ti, torch.tensor(classn, device=device))
                    minibatch.weights.index_put_(
                        ti, torch.tensor(v["weight"], device=device)
                    )

                aug_size += 1

        return aug_size

    def __get_activations_data(self, step):
        aug_size = 0

        minibatch = self.__get_current_augmented_minibatch(step)
        classes_to_sample = self.__get_random_index(
            self.num_representatives_distillation
        )

        for v in classes_to_sample.values():
            for index in v["indices"]:
                ti = (torch.tensor([aug_size]),)
                device = minibatch.device

                # activations
                rehearsal_activations_index = index * self.num_samples_per_activation
                for l in range(0, self.num_samples_per_activation):
                    m = rehearsal_activations_index + l
                    minibatch.activations[l].index_put_(
                        ti, self.rehearsal_activations[m].to(device)
                    )

                # associated representatives
                rehearsal_tensor_index = index * self.num_samples_per_representative
                for r in range(0, self.num_samples_per_representative):
                    k = rehearsal_tensor_index + r
                    minibatch.activations_rep[r].index_put_(
                        ti, self.rehearsal_tensor[k].to(device)
                    )

                aug_size += 1

        return aug_size

    def add_data(
        self,
        samples: List[torch.Tensor],
        targets: torch.Tensor,
        step,
        batch_metrics=None,
        activations=None,
    ):
        """
        Fills the rehearsal buffer with (x, y) pairs, sampled randomly from the
        incoming batch of data. Only `num_candidates` will be added to the
        buffer.

        :param samples: tensor containing a batch of input training samples
        :param targets: tensor containing a batch of targets
        :param step: step number, for logging purposes
        :param activations: tensor containing
        """
        if activations is None:
            activations = []

        if self.high_performance:
            with get_timer(
                "accumulate",
                step,
                batch_metrics=batch_metrics,
                dummy=not measure_performance(step),
            ):
                if self.implementation == "flyweight":
                    self.dsl.accumulate(samples, targets, activations)
                else:
                    next_minibatch = self.__get_next_augmented_minibatch(step)
                    self.dsl.accumulate(
                        samples,
                        targets,
                        activations,
                        next_minibatch.samples,
                        next_minibatch.labels,
                        next_minibatch.weights,
                        next_minibatch.activations,
                        next_minibatch.activations_rep,
                    )
        else:
            for i in range(targets.shape[0]):
                label = targets[i].item()

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
                for r in range(0, self.num_samples_per_representative):
                    k = rehearsal_tensor_index + r
                    self.rehearsal_tensor.index_put_(
                        (torch.tensor([k]),), samples[r][i].cpu()
                    )

                # logits
                rehearsal_activations_index = (
                    self.budget_per_class * label + index
                ) * self.num_samples_per_activation
                for l in range(self.num_samples_per_activation):
                    m = rehearsal_activations_index + l
                    self.rehearsal_activations.index_put_(
                        (torch.tensor([m]),), activations[l][i].float().cpu()
                    )

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
