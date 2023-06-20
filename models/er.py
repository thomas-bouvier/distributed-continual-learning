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


    def before_all_tasks(self, train_data_regime):
        self.buffer = Buffer(train_data_regime.total_num_classes,
            train_data_regime.sample_shape, self.batch_size,
            budget_per_class=self.config.get('rehearsal_size'),
            num_candidates=self.config.get('num_candidates'),
            num_representatives=self.config.get('num_representatives'),
            provider=self.config.get('provider'),
            discover_endpoints=self.config.get('discover_endpoints'),
            cuda=self._is_on_cuda(), cuda_rdma=self.config.get('cuda_rdma'),
            mode=self.config.get('implementation'))

        self.mask = torch.ones(train_data_regime.total_num_classes, device=self._device()).float()
        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')


    def before_every_task(self, task_id, train_data_regime):
        super().before_every_task(task_id, train_data_regime)

        if self.use_mask:
            # Create mask so the loss is only used for classes learnt during this task
            self.mask = torch.tensor(train_data_regime.previous_classes_mask, device=self._device()).float()
            self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

        if task_id > 0:
            self.buffer.enable_augmentation()


    def train_one_step(self, x, y, meters, step, measure_performance=False):
        '''Former nil_cpp implementation

        step: dict containing `task_id`, `epoch` and `batch` keys for logging purposes only
        '''
        w = torch.ones(self.batch_size, device=self._device())

        # Get data from the last iteration (blocking)
        aug_x, aug_y, aug_w = self.buffer.update(x, y, w, step,
                                        measure_performance=measure_performance)

        with get_timer('train', step["batch"]):
            self.optimizer_regime.update(step["epoch"], step["batch"])
            self.optimizer_regime.zero_grad()

            # Forward pass
            with autocast(enabled=self.use_amp):
                output = self.backbone_model(aug_x)
                loss = self.criterion(output, aug_y)

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
            total_weight = hvd.allreduce(torch.sum(aug_w), name='total_weight', op=hvd.Sum)
            dw = aug_w / total_weight * self.batch_size * hvd.size() / self.batch_size

            # Backward pass
            self.scaler.scale(loss).backward(dw)
            self.optimizer_regime.optimizer.synchronize()
            with self.optimizer_regime.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer_regime.optimizer)
                self.scaler.update()

            # Measure accuracy and record metrics
            prec1, prec5 = accuracy(output, aug_y, topk=(1, 5))
            meters["loss"].update(loss.sum() / self.mask[aug_y].sum())
            meters["prec1"].update(prec1, aug_x.size(0))
            meters["prec5"].update(prec5, aug_x.size(0))
            meters["num_samples"].update(aug_x.size(0))
            meters["local_rehearsal_size"].update(self.buffer.get_size())
