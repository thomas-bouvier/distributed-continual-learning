import copy
import horovod.torch as hvd
import logging
import tensorwatch
import time
import torch
import wandb

from torch.utils.tensorboard import SummaryWriter
from utils.meters import AverageMeter, accuracy
from utils.utils import move_cuda, synchronize_cuda


class Agent:
    def __init__(
        self,
        model,
        use_amp,
        config,
        optimizer_regime,
        criterion,
        cuda,
        buffer_cuda,
        log_interval,
        state_dict=None,
    ):
        super(Agent, self).__init__()
        self.model = model
        self.use_amp = use_amp
        self.config = config
        self.optimizer_regime = optimizer_regime
        self.criterion = criterion
        self.cuda = cuda
        self.buffer_cuda = buffer_cuda
        self.log_interval = log_interval
        self.global_epoch = 0
        self.epoch = 0
        self.global_steps = 0
        self.steps = 0
        self.minimal_eval_loss = float("inf")
        self.best_model = None
        self.writer = None
        self.writer_images = False
        self.watcher = None
        self.streams = {}

        if self.use_amp:
            try:
                global amp
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to run this app.")

        if state_dict is not None:
            self.model.load_state_dict(state_dict)
            self.initial_snapshot = copy.deepcopy(state_dict)
        else:
            self.initial_snapshot = copy.deepcopy(self.model.state_dict())

        self.epoch_load_time = 0
        self.epoch_move_time = 0
        self.last_batch_load_time = 0
        self.last_batch_move_time = 0

    """
    Forward pass for the current epoch
    """

    def loop(self, data_regime, training=False):
        prefix = "train" if training else "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5"]
        }
        epoch_time = 0
        step_count = 0

        start_batch_time = time.time()
        start_load_time = start_batch_time
        for i_batch, (x, y, t) in enumerate(data_regime.get_loader()):
            #torch.cuda.nvtx.range_push(f"Batch {i_batch}")
            synchronize_cuda(self.cuda)
            self.last_batch_load_time = time.time() - start_load_time
            self.epoch_load_time += self.last_batch_load_time

            output, loss = self._step(
                i_batch, x, y, training=training
            )
            synchronize_cuda(self.cuda)
            batch_time = time.time() - start_batch_time
            epoch_time += batch_time

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            meters["loss"].update(loss)
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))

            if i_batch % self.log_interval == 0 or i_batch == len(
                data_regime.get_loader()
            ):
                logging.info(
                    "{phase}: epoch: {0} [{1}/{2}]\t"
                    "Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t"
                    "Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t"
                    "Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t".format(
                        self.epoch,
                        i_batch,
                        len(data_regime.get_loader()),
                        phase="TRAINING" if training else "EVALUATING",
                        meters=meters,
                    )
                )

                logging.info(f"batch {i_batch} time {batch_time} sec")
                logging.info(
                    f"\tbatch load time {self.last_batch_load_time} sec ({self.last_batch_load_time*100/batch_time}%)")
                logging.info(
                    f"\tbatch move time {self.last_batch_move_time} sec ({self.last_batch_move_time*100/batch_time}%)")

                if hvd.rank() == 0 and hvd.local_rank() == 0:
                    wandb.log({f"{prefix}_loss": meters["loss"].avg,
                               "step": self.global_steps,
                               "epoch": self.global_epoch,
                               "batch": i_batch,
                               f"{prefix}_prec1": meters["prec1"].avg,
                               f"{prefix}_prec5": meters["prec5"].avg})
                    if training:
                        wandb.log({"lr": self.optimizer_regime.get_lr()[0],
                                   "step": self.global_steps,
                                   "epoch": self.global_epoch})
                if self.writer is not None:
                    self.writer.add_scalar(
                        f"{prefix}_loss", meters["loss"].avg, self.global_steps
                    )
                    self.writer.add_scalar(
                        f"{prefix}_prec1",
                        meters["prec1"].avg,
                        self.global_steps,
                    )
                    if training:
                        self.writer.add_scalar(
                            "lr", self.optimizer_regime.get_lr()[
                                0], self.global_steps
                        )
                    self.writer.flush()
                if self.watcher is not None:
                    self.observe(
                        trainer=self,
                        model=self.model,
                        optimizer=self.optimizer_regime.optimizer,
                        data=(x, y),
                    )
                    self.stream_meters(meters, prefix=prefix)
                    if training:
                        self.write_stream(
                            "lr",
                            (self.global_steps,
                             self.optimizer_regime.get_lr()[0]),
                        )
            # torch.cuda.nvtx.range_pop()
            step_count += 1
            synchronize_cuda(self.cuda)
            start_batch_time = time.time()
            start_load_time = start_batch_time

        if training:
            self.global_epoch += 1
            logging.info(f"\nCUMULATED VALUES:")
            logging.info(f"epoch time {epoch_time} sec")
            logging.info(
                f"\tepoch load time {self.epoch_load_time} sec ({self.epoch_load_time*100/epoch_time}%)")
            logging.info(
                f"\tepoch move time {self.epoch_move_time} sec ({self.epoch_move_time*100/epoch_time}%)")
        self.epoch_load_time = 0
        self.epoch_move_time = 0

        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = epoch_time
        meters["step_count"] = step_count

        return meters

    def _step(
        self,
        i_batch,
        inputs,
        target,
        training=False,
        chunk_batch=1,
    ):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer_regime.zero_grad()
            self.optimizer_regime.update(self.epoch, self.steps)

        for i, (x, y) in enumerate(zip(inputs.chunk(chunk_batch, dim=0),
                                       target.chunk(chunk_batch, dim=0))):
            #torch.cuda.nvtx.range_push("Copy to device")
            start_move_time = time.time()
            x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)
            self.last_batch_move_time = time.time() - start_move_time
            self.epoch_move_time += self.last_batch_move_time
            # torch.cuda.nvtx.range_pop()

            #torch.cuda.nvtx.range_push("Forward pass")
            output = self.model(x)
            loss = self.criterion(output, y)
            # torch.cuda.nvtx.range_pop()

        if training:
            #torch.cuda.nvtx.range_push("Optimizer step")
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer_regime.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.optimizer_regime.step()
            else:
                loss.backward()
                self.optimizer_regime.step()
            # torch.cuda.nvtx.range_pop()
            self.global_steps += 1
            self.steps += 1

        return output, loss

    def train(self, data_regime):
        # switch to train mode
        self.model.train()
        self.write_stream("epoch", (self.global_steps, self.epoch))
        return self.loop(data_regime, training=True)

    def validate(self, data_regime):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.loop(data_regime, training=False)

    def before_all_tasks(self, train_data_regime):
        self.num_classes = train_data_regime.num_classes

    def after_all_tasks(self):
        pass

    def before_every_task(self, task_id, train_data_regime):
        self.steps = 0

        # Distribute the data
        #torch.cuda.nvtx.range_push("Distribute dataset")
        train_data_regime.get_loader(True)
        # torch.cuda.nvtx.range_pop()

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

    def after_every_task(self):
        pass

    def after_every_epoch(self):
        pass

    def set_tensorboard_writer(self, save_path, dummy=False, images=False):
        if dummy:
            return False
        if images:
            self.writer_images = images
        self.writer = SummaryWriter(log_dir=save_path)
        return True

    def set_tensorwatch_watcher(self, filename, port=0, dummy=False):
        if dummy:
            return False
        self.watcher = tensorwatch.Watcher(filename=filename, port=port)
        self.get_stream("train_loss")
        self.get_stream("val_loss")
        self.get_stream("train_prec1")
        self.get_stream("val_prec1")
        self.get_stream("lr")
        self.watcher.make_notebook()
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
            stream.write((self.global_steps, value))
        return True

    def write_stream(self, name, values):
        stream = self.get_stream(name)
        if stream is not None:
            stream.write(values)


def base(model, use_amp, agent_config, optimizer_regime, criterion, cuda, buffer_cuda, log_interval):
    return Agent(
        model,
        use_amp,
        agent_config,
        optimizer_regime,
        criterion,
        cuda,
        buffer_cuda,
        log_interval,
    )
