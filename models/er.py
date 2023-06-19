import copy
import horovod.torch as hvd
import logging
import torch
import torch.nn as nn

from torch.cuda.amp import autocast

from modules import ContinualLearner, Buffer
from utils.meters import get_timer, accuracy
from utils.log import PerformanceResultsLog


class Er(ContinualLearner):
    '''Model for classifying images, "enriched" as ContinualLearner object.'''

    def __init__(
        self,
        backbone_model,
        use_mask,
        use_amp,
        config,
        optimizer_regime,
        batch_size,
        log_buffer,
        log_interval,
        batch_metrics=None,
        state_dict=None,
    ):
        super(Er, self).__init__(
            backbone_model,
            use_amp,
            optimizer_regime,
            batch_size,
            batch_metrics,
            state_dict,
        )

        self.use_mask = use_mask
        self.config = config
        self.log_buffer = log_buffer
        self.log_interval = log_interval

        self.use_memory_buffer = True


    def before_all_tasks(self, train_data_regime):
        self.buffer = Buffer(train_data_regime.total_num_classes, train_data_regime.sample_shape, self.batch_size,
            self.config.get('rehearsal_size'), self.config.get('num_candidates'),
            self.config.get('num_representatives'), self.config.get('provider'),
            self.config.get('discover_endpoints'), self.config.get('cuda_rdma'))

        self.mask = torch.ones(train_data_regime.total_num_classes, device=self._device()).float()
        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')


    def before_every_task(self, task_id, train_data_regime):
        super().before_every_task(task_id, train_data_regime)

        if self.use_mask:
            # Create mask so the loss is only used for classes learnt during this task
            self.mask = torch.tensor(train_data_regime.previous_classes_mask, device=self._device()).float()
            self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

        if self.use_memory_buffer and task_id > 0:
            self.buffer.dsl.enable_augmentation(True)


    def train_one_step(self, x, y, meters, step, measure_performance=False):
        '''Former nil_cpp implementation

        batch: batch number, for logging purposes only
        '''
        aug_size = self.batch_size
        w = torch.ones(self.batch_size, device=self._device())

        if self.use_memory_buffer and step["task_id"] > 0:
            # Get the representatives
            with get_timer('wait', step["batch"], previous_iteration=True):
                aug_size = self.buffer.dsl.wait()
                logging.debug(f"Received {n} samples from other nodes")

                if measure_performance:
                    cpp_metrics = self.buffer.dsl.get_metrics(step["batch"])
                    self.perf_metrics.add(step["batch"]-1, cpp_metrics)

            # Assemble the minibatch
            with get_timer('assemble', step["batch"]):
                current_minibatch = self.get_current_augmented_minibatch()
                x = current_minibatch.x[:aug_size]
                y = current_minibatch.y[:aug_size]
                w = current_minibatch.w[:aug_size]

            """
            if self.log_buffer and step["batch"] % self.log_interval == 0 and hvd.rank() == 0:
                repr_size = self.aug_size - self.batch_size
                if repr_size > 0:
                    captions = []
                    for label, weight in zip(y[-repr_size:], w[-repr_size:]):
                        captions.append(f"y={label.item()} w={weight.item()}")
                    display(f"aug_batch_{step["task_id"]}_{step["epoch"]}_{step["batch"]}", x[-repr_size:], captions=captions)
            """

            # In-advance preparation of next minibatch
            with get_timer('accumulate', step["batch"]):
                next_minibatch = self.get_next_augmented_minibatch()
                self.buffer.dsl.accumulate(
                    x, y, next_minibatch.x, next_minibatch.y, next_minibatch.w
                )

        # Train
        with get_timer('train', step["batch"]):
            self.optimizer_regime.update(step["epoch"], step["batch"])
            self.optimizer_regime.zero_grad()

            # Forward pass
            with autocast(enabled=self.use_amp):
                output = self.backbone_model(x)
                loss = self.criterion(output, y)

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
            total_weight = hvd.allreduce(torch.sum(w), name='total_weight', op=hvd.Sum)
            dw = w / total_weight * self.batch_size * hvd.size() / self.batch_size

            # Backward pass
            self.scaler.scale(loss).backward(dw)
            self.optimizer_regime.optimizer.synchronize()
            with self.optimizer_regime.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer_regime.optimizer)
                self.scaler.update()

            # Measure accuracy and record metrics
            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            meters["loss"].update(loss.sum() / self.mask[y].sum())
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))
            meters["num_samples"].update(aug_size)
            meters["local_rehearsal_size"].update(self.buffer.dsl.get_rehearsal_size())


    """
    def train_one_step(self, x, y, meters, step, measure_performance=False):
        '''Former nil_cpp_cat implementation'''

        # Get the representatives
        with get_timer('wait', step["batch"], previous_iteration=True):
            self.repr_size = self.buffer.dsl.wait()
            n = self.repr_size
            if n > 0:
                logging.debug(f"Received {n} samples from other nodes")

            if measure_performance:
                cpp_metrics = self.buffer.dsl.get_metrics(step["batch"])
                self.perf_metrics.add(step["batch"]-1, cpp_metrics)

        # Assemble the minibatch
        with get_timer('assemble', step["batch"]):
            w = torch.ones(self.batch_size, device=self._device())
            new_x = torch.cat((x, self.next_minibatch.x[:self.repr_size]))
            new_y = torch.cat((y, self.next_minibatch.y[:self.repr_size]))
            new_w = torch.cat((w, self.next_minibatch.w[:self.repr_size]))

        if self.log_buffer and step["batch"] % self.log_interval == 0 and hvd.rank() == 0:
            if self.repr_size > 0:
                captions = []
                for y, w in zip(self.next_minibatch.y[-self.repr_size:], self.next_minibatch.w[-self.repr_size:]):
                    captions.append(f"y={y.item()} w={w.item()}")
                display(f"aug_batch_{step["task_id"]}_{step["epoch"]}_{step["batch"]}", self.next_minibatch.x[-self.repr_size:], captions=captions)

        # In-advance preparation of next minibatch
        with get_timer('accumulate', step["batch"]):
            self.buffer.dsl.accumulate(x, y)

        # Train
        with get_timer('train', step["batch"]):
            self.optimizer_regime.update(step["epoch"], step["batch"])
            self.optimizer_regime.zero_grad()

            with autocast(enabled=self.use_amp):
                output = self.backbone_model(new_x)
                loss = self.criterion(output, new_y)

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # https://stackoverflow.com/questions/43451125/pytorch-what-a-re-the-gradient-arguments
            total_weight = hvd.allreduce(torch.sum(new_w), name='total_weight', op=hvd.Sum)
            dw = new_w / total_weight * self.batch_size * hvd.size() / self.batch_size

            self.scaler.scale(loss).backward(dw)
            self.optimizer_regime.optimizer.synchronize()
            with self.optimizer_regime.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer_regime.optimizer)
                self.scaler.update()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, new_y, topk=(1, 5))
            # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
            meters["loss"].update(loss.sum() / self.mask[new_y].sum())
            meters["prec1"].update(prec1, new_x.size(0))
            meters["prec5"].update(prec5, new_x.size(0))
            meters["num_samples"].update(self.batch_size + self.repr_size)
            meters["local_rehearsal_size"].update(self.buffer.dsl.get_rehearsal_size())
    """
