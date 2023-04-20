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
    def validate(self, data_regime):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.validate_one_epoch(data_regime)

    def before_all_tasks(self, train_data_regime):
        self.mask = torch.ones(train_data_regime.total_num_classes, device=self.device).float()
        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

    def before_every_task(self, task_id, train_data_regime):
        self.task_id = task_id
        self.batch = 0

        # Distribute the data
        train_data_regime.get_loader(True)

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

    """
    Forward pass for the current epoch
    """
    def train_one_epoch(self, data_regime):
        prefix = "train"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5", "num_samples"]
        }
        epoch_time = 0
        last_batch_time = 0
        loader = data_regime.get_loader()

        enable_tqdm = self.log_level in ('info') and hvd.rank() == 0
        with tqdm(total=len(loader),
              desc=f"{prefix} epoch #{self.epoch + 1}",
              disable=not enable_tqdm
        ) as progress:
            timer = self.get_timer('load')
            timer.__enter__()
            start_batch_time = time.perf_counter()

            for x, y, t in loader:
                x, y = move_cuda(x, self.cuda), move_cuda(y.long(), self.cuda)
                timer.__exit__(None, None, None)

                self.train_one_step(x, y, meters)

                last_batch_time = time.perf_counter() - start_batch_time
                epoch_time += last_batch_time

                if hvd.rank() == 0:
                    # Performance metrics
                    if self.measure_performance() and self.batch_metrics is not None:
                        metrics = self.perf_metrics.get(self.batch)
                        batch_metrics_values = dict(
                            epoch=self.epoch,
                            batch=self.batch,
                            time=last_batch_time,
                            load_time=metrics.get('load', 0),
                            train_time=metrics.get('train', 0)
                        )
                        self.batch_metrics.add(**batch_metrics_values)
                        self.batch_metrics.save()

                if self.batch % self.log_interval == 0 or self.batch == len(loader):
                    logging.debug(
                        "{0}: epoch: {1} [{2}/{3}]\t"
                        "Loss {meters[loss].avg:.4f}\t"
                        "Prec@1 {meters[prec1].avg:.3f}\t"
                        "Prec@5 {meters[prec5].avg:.3f}\t".format(
                            prefix,
                            self.epoch + 1,
                            self.batch %len(loader) + 1,
                            len(loader),
                            meters=meters,
                        )
                    )

                    if self.measure_performance():
                        metrics = self.perf_metrics.get(self.batch)
                        logging.debug(f"batch {self.batch} time {last_batch_time} sec")
                        logging.debug(
                            f"\t[Python] batch load time {metrics.get('load', 0)} sec ({metrics.get('load', 0)*100/last_batch_time}%)")
                        logging.debug(
                            f"\t[Python] batch train time {metrics.get('train', 0)} sec ({metrics.get('train', 0)*100/last_batch_time}%)")

                progress.set_postfix({'loss': meters["loss"].avg.item(),
                            'accuracy': meters["prec1"].avg.item()})
                progress.update(1)

                self.global_batch += 1
                self.batch += 1

                timer = self.get_timer('load')
                timer.__enter__()
                start_batch_time = time.perf_counter()

        if hvd.rank() == 0:
            # Performance metrics
            wandb.log({"epoch": self.global_epoch,
                    f"{prefix}_loss": meters["loss"].avg,
                    f"{prefix}_prec1": meters["prec1"].avg,
                    f"{prefix}_prec5": meters["prec5"].avg,
                    "lr": self.optimizer_regime.get_lr()[0]})

        self.global_epoch += 1
        self.epoch += 1

        logging.info(f"\nCUMULATED VALUES:")
        logging.info(f"epoch time {epoch_time} sec")
        """
        logging.info(
            f"\tepoch load time {self.epoch_load_time} sec ({self.epoch_load_time*100/epoch_time}%)")
        logging.info(
            f"\tepoch train time {self.epoch_train_time} sec ({self.epoch_train_time*100/epoch_time}%)")
        """

        num_samples = meters["num_samples"].sum.item()
        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["num_samples"] = num_samples
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = epoch_time
        meters["batch"] = self.batch

        return meters

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

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            if self.use_amp:
                self.scaler.scale(loss).backward(torch.ones_like(loss))
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward(torch.ones_like(loss))
                self.optimizer_regime.step()

            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            meters["loss"].update(loss.sum() / self.mask[y].sum(), x.size(0))
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))
            meters["num_samples"].update(self.batch_size)

    def validate_one_epoch(self, data_regime):
        prefix = "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5", "num_samples"]
        }
        epoch_time = 0
        last_batch_time = 0
        loader = data_regime.get_loader()

        criterion = torch.nn.CrossEntropyLoss()

        enable_tqdm = self.log_level in ('info') and hvd.rank() == 0
        with tqdm(total=len(loader),
              desc=f"Task #{self.task_id + 1} {prefix} epoch #{self.epoch + 1}",
              disable=not enable_tqdm
        ) as progress:
            start_batch_time = time.perf_counter()

            for x, y, t in loader:
                x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)

                if self.use_amp:
                    with autocast():
                        output = self.model(x)
                        loss = criterion(output, y)
                else:
                    output = self.model(x)
                    loss = criterion(output, y)

                prec1, prec5 = accuracy(output, y, topk=(1, 5))
                meters["loss"].update(loss, x.size(0))
                meters["prec1"].update(prec1, x.size(0))
                meters["prec5"].update(prec5, x.size(0))

                last_batch_time = time.perf_counter() - start_batch_time
                epoch_time += last_batch_time

                progress.set_postfix({'loss': meters["loss"].avg.item(),
                           'accuracy': meters["prec1"].avg.item()})
                progress.update(1)

        if hvd.rank() == 0:
            wandb.log({"epoch": self.global_epoch,
                    f"{prefix}_loss": meters["loss"].avg,
                    f"{prefix}_prec1": meters["prec1"].avg,
                    f"{prefix}_prec5": meters["prec5"].avg,
                    "lr": self.optimizer_regime.get_lr()[0]})

        logging.info(f"epoch time {epoch_time} sec")

        num_samples = meters["num_samples"].sum.item()
        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["num_samples"] = num_samples
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = epoch_time
        meters["batch"] = self.batch

        return meters

    def after_all_tasks(self):
        pass

    def after_every_task(self):
        pass

    def after_every_epoch(self):
        pass

    def set_tensorboard_writer(self, save_path, dummy=False, images=False):
        if dummy:
            return False
        if images:
            self.writer_images = images
        try:
            global SummaryWriter
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                "Please install tensorboard to run this app.")
        self.writer = SummaryWriter(log_dir=save_path)
        return True

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
