import copy
import horovod.torch as hvd
import logging
import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast
from utils.meters import AverageMeter, MeasureTime, get_timer, accuracy
from utils.log import PerformanceResultsLog, display
from utils.utils import move_cuda

from modules import ContinualLearner, Buffer

class Er(ContinualLearner):
    '''Model for classifying images, "enriched" as ContinualLearner- and Buffer-object.'''

    def __init__(
        self,
        backbone_model,
        use_mask,
        use_amp,
        config,
        optimizer_regime,
        batch_size,
        cuda,
        log_level,
        log_buffer,
        log_interval,
        batch_metrics=None,
        state_dict=None,
    ):
        super().__init__()

        self.backbone_model = backbone_model
        self.use_mask = use_mask
        self.use_amp = use_amp
        self.config = config
        self.optimizer_regime = optimizer_regime
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = 'cuda' if self.cuda else 'cpu'
        self.log_level = log_level
        self.log_buffer = log_buffer
        self.log_interval = log_interval
        self.batch_metrics = batch_metrics

        self.global_epoch = 0
        self.epoch = 0
        self.global_batch = 0
        self.batch = 0
        self.minimal_eval_loss = float("inf")
        self.best_model = None

        self.use_memory_buffer = True
        self.current_rehearsal_size = 0

        if self.use_amp:
            self.scaler = GradScaler()

        if state_dict is not None:
            self.backbone_model.load_state_dict(state_dict)
            self.initial_snapshot = copy.deepcopy(state_dict)
        else:
            self.initial_snapshot = copy.deepcopy(self.backbone_model.state_dict())

        self.perf_metrics = PerformanceResultsLog()


    def before_all_tasks(self, train_data_regime):
        self.buffer = Buffer(train_data_regime.total_num_classes, train_data_regime.sample_shape, self.batch_size,
            self.config.get('rehearsal_size'), self.config.get('num_candidates'),
            self.config.get('num_representatives'), self.config.get('provider'),
            self.config.get('discover_endpoints'), self.config.get('cuda_rdma'))

        self.mask = torch.ones(train_data_regime.total_num_classes, device=self.device).float()
        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')


    def before_every_task(self, task_id, train_data_regime):
        self.task_id = task_id
        self.batch = 0

        # Distribute the data
        train_data_regime.get_loader(task_id)

        if self.best_model is not None:
            logging.debug(
                f"Loading best model with minimal eval loss ({self.minimal_eval_loss}).."
            )
            self.backbone_model.load_state_dict(self.best_model)
            self.minimal_eval_loss = float("inf")

        if task_id > 0:
            if self.config.get("reset_state_dict", False):
                logging.debug("Resetting model internal state..")
                self.backbone_model.load_state_dict(
                    copy.deepcopy(self.initial_snapshot))
            self.optimizer_regime.reset(self.backbone_model.parameters())

        if self.use_mask:
            # Create mask so the loss is only used for classes learnt during this task
            self.mask = torch.tensor(train_data_regime.previous_classes_mask, device=self.device).float()
            self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

        if self.use_memory_buffer and task_id > 0:
            self.buffer.dsl.enable_augmentation(True)


    def train_one_step(self, x, y, meters):
        '''Former nil_cpp implementation'''

        aug_size = self.batch_size
        w = torch.ones(self.batch_size, device=self.device)

        if self.use_memory_buffer and self.task_id > 0:
            # Get the representatives
            with get_timer('wait', self.batch, previous_iteration=True):
                aug_size = self.buffer.dsl.wait()
                logging.debug(f"Received {n} samples from other nodes")

                if self.measure_performance():
                    cpp_metrics = self.buffer.dsl.get_metrics(self.batch)
                    self.perf_metrics.add(self.batch-1, cpp_metrics)

            # Assemble the minibatch
            with get_timer('assemble', self.batch):
                current_minibatch = self.get_current_augmented_minibatch()
                x = current_minibatch.x[:aug_size]
                y = current_minibatch.y[:aug_size]
                w = current_minibatch.w[:aug_size]

            if self.log_buffer and self.batch % self.log_interval == 0 and hvd.rank() == 0:
                repr_size = self.aug_size - self.batch_size
                if repr_size > 0:
                    captions = []
                    for label, weight in zip(y[-repr_size:], w[-repr_size:]):
                        captions.append(f"y={label.item()} w={weight.item()}")
                    display(f"aug_batch_{self.task_id}_{self.epoch}_{self.batch}", x[-repr_size:], captions=captions)

            # In-advance preparation of next minibatch
            with get_timer('accumulate', self.batch):
                next_minibatch = self.get_next_augmented_minibatch()
                self.buffer.dsl.accumulate(
                    x, y, next_minibatch.x, next_minibatch.y, next_minibatch.w
                )

        # Train
        with get_timer('train', self.batch):
            self.optimizer_regime.update(self.epoch, self.batch)
            self.optimizer_regime.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.backbone_model(x)
                    loss = self.criterion(output, y)
            else:
                output = self.backbone_model(x)
                loss = self.criterion(output, y)

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
            total_weight = hvd.allreduce(torch.sum(w), name='total_weight', op=hvd.Sum)
            dw = w / total_weight * self.batch_size * hvd.size() / self.batch_size

            if self.use_amp:
                self.scaler.scale(loss).backward(dw)
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward(dw)
                self.optimizer_regime.step()

            self.current_rehearsal_size = self.buffer.dsl.get_rehearsal_size()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            meters["loss"].update(loss.sum() / self.mask[y].sum())
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))
            meters["num_samples"].update(aug_size)
            meters["local_rehearsal_size"].update(self.current_rehearsal_size)


    """
    def train_one_step(self, x, y, meters):
        '''Former nil_cpp_cat implementation'''

        # Get the representatives
        with get_timer('wait', self.batch, previous_iteration=True):
            self.repr_size = self.buffer.dsl.wait()
            n = self.repr_size
            if n > 0:
                logging.debug(f"Received {n} samples from other nodes")

            if self.measure_performance():
                cpp_metrics = self.buffer.dsl.get_metrics(self.batch)
                self.perf_metrics.add(self.batch-1, cpp_metrics)

        # Assemble the minibatch
        with get_timer('assemble', self.batch):
            w = torch.ones(self.batch_size, device=self.device)
            new_x = torch.cat((x, self.next_minibatch.x[:self.repr_size]))
            new_y = torch.cat((y, self.next_minibatch.y[:self.repr_size]))
            new_w = torch.cat((w, self.next_minibatch.w[:self.repr_size]))

        if self.log_buffer and self.batch % self.log_interval == 0 and hvd.rank() == 0:
            if self.repr_size > 0:
                captions = []
                for y, w in zip(self.next_minibatch.y[-self.repr_size:], self.next_minibatch.w[-self.repr_size:]):
                    captions.append(f"y={y.item()} w={w.item()}")
                display(f"aug_batch_{self.task_id}_{self.epoch}_{self.batch}", self.next_minibatch.x[-self.repr_size:], captions=captions)

        # In-advance preparation of next minibatch
        with get_timer('accumulate', self.batch):
            self.buffer.dsl.accumulate(x, y)

        # Train
        with get_timer('train', self.batch):
            self.optimizer_regime.update(self.epoch, self.batch)
            self.optimizer_regime.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.backbone_model(new_x)
                    loss = self.criterion(output, new_y)
            else:
                output = self.backbone_model(new_x)
                loss = self.criterion(output, new_y)

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # https://stackoverflow.com/questions/43451125/pytorch-what-a-re-the-gradient-arguments
            total_weight = hvd.allreduce(torch.sum(new_w), name='total_weight', op=hvd.Sum)
            dw = new_w / total_weight * self.batch_size * hvd.size() / self.batch_size

            if self.use_amp:
                self.scaler.scale(loss).backward(dw)
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward(dw)
                self.optimizer_regime.step()

            self.current_rehearsal_size = self.buffer.dsl.get_rehearsal_size()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, new_y, topk=(1, 5))
            # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
            meters["loss"].update(loss.sum() / self.mask[new_y].sum())
            meters["prec1"].update(prec1, new_x.size(0))
            meters["prec5"].update(prec5, new_x.size(0))
            meters["num_samples"].update(self.batch_size + self.repr_size)
            meters["local_rehearsal_size"].update(self.current_rehearsal_size)
    """


    def measure_performance(self):
        #assert self.current_rehearsal_size > 0
        return self.task_id == 1 and self.epoch == 10
