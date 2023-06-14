import copy
import horovod.torch as hvd
import logging
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb

from torch.cuda.amp import GradScaler, autocast
from utils.log import PerformanceResultsLog
from utils.meters import AverageMeter, MeasureTime, accuracy
from utils.utils import move_cuda, display


class Agent:
    def __init__(
        self,
        model,
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
        super(Agent, self).__init__()
        self.model = model
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
        self.writer = None
        self.writer_images = False
        self.watcher = None
        self.streams = {}

        if self.use_amp:
            self.scaler = GradScaler()

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            self.initial_snapshot = copy.deepcopy(state_dict)
        else:
            self.initial_snapshot = copy.deepcopy(self.model.state_dict())

        self.perf_metrics = PerformanceResultsLog()

    def measure_performance(self):
        return self.task_id == 0 and self.global_epoch == 0

    """Train for one epoch"""
    def train(self, data_regime):
        # switch to train mode
        self.model.train()
        self.write_stream("epoch", (self.global_batch, self.epoch))
        return self.train_one_epoch(data_regime)

    """Validate on one epoch"""
    def validate(self, data_regime, task_id):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.validate_one_epoch(data_regime, task_id)

    def before_all_tasks(self, train_data_regime):
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
            self.model.load_state_dict(self.best_model)
            self.minimal_eval_loss = float("inf")

        if task_id > 0:
            if self.config.get("reset_state_dict", False):
                logging.debug("Resetting model internal state..")
                self.model.load_state_dict(
                    copy.deepcopy(self.initial_snapshot))
            self.optimizer_regime.reset(self.model.parameters())

        if self.use_mask:
            # Create mask so the loss is only used for classes learnt during this task
            self.mask = torch.tensor(train_data_regime.previous_classes_mask, device=self.device).float()
            self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')


    def train_one_step(
        self,
        x,
        y,
        meters,
    ):
        # Train
        with self.get_timer('train'):
            self.optimizer_regime.update(self.epoch, self.batch)
            self.optimizer_regime.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(x)
                    loss = self.criterion(output, y)
            else:
                output = self.model(x)
                loss = self.criterion(output, y)
            loss = loss.sum() / self.mask[y].sum()

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                self.optimizer_regime.step()

            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            meters["loss"].update(loss)
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))
            meters["num_samples"].update(self.batch_size)


    def after_all_tasks(self):
        pass

    def after_every_task(self):
        pass

    def after_every_epoch(self):
        pass


    def get_state_dict(self):
        return self.model.state_dict()

    def get_stream(self, name, **kwargs):
        if self.watcher is None:
            return None
        if name not in self.streams.keys():
            self.streams[name] = self.watcher.create_stream(
                name=name, **kwargs)
        return self.streams[name]

    def observe(self, **kwargs):
        if self.watcher is None:
            return False
        self.watcher.observe(**kwargs)
        return True

    def stream_meters(self, meters_dict, prefix=None):
        if self.watcher is None:
            return False
        for name, value in meters_dict.items():
            if prefix is not None:
                name = "_".join([prefix, name])
            value = value.val
            stream = self.get_stream(name)
            if stream is None:
                continue
            stream.write((self.global_batch, value))
        return True

    def write_stream(self, name, values):
        stream = self.get_stream(name)
        if stream is not None:
            stream.write(values)

    def get_timer(self, name):
        return MeasureTime(self.batch, name, self.perf_metrics, self.cuda, not self.measure_performance())

def base(model, use_mask, use_amp, agent_config, optimizer_regime, batch_size, cuda, log_level, log_buffer, log_interval, batch_metrics):
    return Agent(
        model,
        use_mask,
        use_amp,
        agent_config,
        optimizer_regime,
        batch_size,
        cuda,
        log_level,
        log_buffer,
        log_interval,
        batch_metrics
    )
