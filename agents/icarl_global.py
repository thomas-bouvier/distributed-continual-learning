import copy
import horovod.torch as hvd
import logging
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import wandb

from agents.base import Agent
from torch.cuda.amp import GradScaler, autocast
from utils.utils import move_cuda
from utils.meters import AverageMeter, accuracy


def make_candidates(n, mem_x, mem_y, cand_x, cand_y, lock_make, lock_made, num_batches):
    for i in range(num_batches):
        lock_make.acquire()
        selection = torch.randperm(len(mem_x))[n].clone()
        nx = mem_x[selection].clone() - cand_x
        ny = mem_y[selection].clone() - cand_y
        cand_x += nx
        cand_y += ny
        lock_made.release()


class icarl_v1_agent(Agent):
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
        super(icarl_v1_agent, self).__init__(
            model,
            use_amp,
            config,
            optimizer_regime,
            criterion,
            cuda,
            buffer_cuda,
            log_interval,
            state_dict,
        )

        if self.use_amp:
            self.scaler = GradScaler()

        self.num_exemplars = 0
        self.num_candidates = config.get("num_candidates", 20)

        # memory
        self.buf_x = None  # stores raw inputs, PxD
        self.buf_y = None
        self.mem_class_x = {}  # stores exemplars class by class
        self.mem_class_y = {}
        self.mem_class_means = {}

        # setup distillation losses
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.lsm = nn.LogSoftmax(dim=1)
        self.sm = nn.Softmax(dim=1)

    def before_all_tasks(self, train_data_regime):
        super().before_all_tasks(train_data_regime)
        self.memory_size = config.get(
            "num_representatives", 6000) * self.num_classes

    def before_every_task(self, task_id, train_data_regime):
        self.steps = 0

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

        # Create mask so the loss is only used for classes learnt during this task
        self.nc = set([data[1] for data in train_data_regime.get_data()])
        mask = torch.tensor([False for _ in range(self.num_classes)])
        for y in self.nc:
            mask[y] = True
        self.mask = move_cuda(mask.float(), self.cuda)

        self.criterion = nn.CrossEntropyLoss(weight=self.mask)

        # Distillation
        self.should_distill = self.mem_class_x != {}
        if self.should_distill:
            self.mem_x = torch.cat(
                [samples.cpu() for samples in self.mem_class_x.values()]
            ).share_memory_()
            self.mem_y = torch.cat(
                [targets.cpu() for targets in self.mem_class_y.values()]
            ).share_memory_()

    def after_every_task(self):
        self.update_exemplars(self.nc)

    """
    Forward pass for the current epoch
    """

    def loop(self, data_regime, training=False):
        prefix = "train" if training else "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5"]
        }
        start = time.time()
        step_count = 0

        # Distillation, increment taskset
        if self.should_distill and training:
            self.cand_x = torch.zeros(
                [self.num_candidates] + list(self.mem_x[0].size())
            ).share_memory_()
            self.cand_y = torch.zeros(
                [self.num_candidates] + list(self.mem_y[0].size())
            ).share_memory_()

            self.lock_make = mp.Lock()
            self.lock_made = mp.Lock()
            self.lock_made.acquire()

            self.p = mp.Process(
                target=make_candidates,
                args=(
                    self.num_candidates,
                    self.mem_x,
                    self.mem_y,
                    self.cand_x,
                    self.cand_y,
                    self.lock_make,
                    self.lock_made,
                    len(data_regime.get_loader()),
                ),
            )
            self.p.start()

        for i_batch, (x, y, t) in enumerate(data_regime.get_loader()):
            batch_start = time.time()
            output, loss = self._step(
                i_batch, x, y, training=training
            )
            batch_end = time.time()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output[: y.size(0)], y, topk=(1, 5))
            meters["loss"].update(loss, x.size(0))
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
                logging.info(
                    f"Time taken for batch {i_batch} is {batch_end - batch_start} sec")

                if hvd.rank() == 0:
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
                    self.writer.add_scalar(
                        f"{prefix}_prec5",
                        meters["prec5"].avg,
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
            step_count += 1

        # Distillation
        if self.should_distill:
            self.p.join()
        end = time.time()

        if training:
            self.global_epoch += 1

        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = end - start
        meters["step_count"] = step_count

        return meters

    def _step(
        self,
        i_batch,
        inputs,
        target,
        training=False,
        distill=False,
    ):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer_regime.zero_grad()
            self.optimizer_regime.update(self.epoch, self.steps)

        x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)

        if training:
            if self.buf_x is None:
                self.buf_x = x.detach().cpu()
                self.buf_y = y.detach().cpu()
            else:
                self.buf_x = torch.cat((self.buf_x, x.detach().cpu()))
                self.buf_y = torch.cat((self.buf_y, y.detach().cpu()))

            # Distillation
            if self.should_distill:
                self.lock_made.acquire()
                x = torch.cat((x, move_cuda(self.cand_x, self.cuda)))
                dist_y = move_cuda(self.cand_y, self.cuda)
                self.lock_make.release()

            if self.use_amp:
                with autocast(dtype=torch.float16):
                    output = self.model(x)
                    loss = self.criterion(output[: y.size(0)], y)
            else:
                output = self.model(x)
                loss = self.criterion(output[: y.size(0)], y)

            # Compute distillation loss
            if self.should_distill:
                loss += self.kl(self.lsm(output[y.size(0):]),
                                self.sm(dist_y))

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                self.optimizer_regime.step()
            self.global_steps += 1
            self.steps += 1
        else:
            output = self.forward(x)
            loss = self.criterion(output[: y.size(0)], y)

        return output, loss

    def after_every_epoch(self):
        if self.epoch + 1 != self.epochs:
            self.buf_x = None
            self.buf_y = None

    def forward(self, x):
        self.model.eval()
        ns = x.size(0)

        with torch.no_grad():
            classpred = torch.LongTensor(ns)
            self.model(x)
            preds = move_cuda(
                self.model.feature_vector.detach().clone(), self.cuda)
            mean_features = move_cuda(self.mean_features, self.cuda)
            dist = torch.cdist(
                preds.view(1, *preds.size()),
                mean_features.view(1, *mean_features.size()),
            ).view(ns, len(self.mem_class_means.keys()))

            for ss in range(ns):
                classpred[ss] = torch.argmin(dist[ss])

            out = move_cuda(torch.zeros(ns, self.num_classes), self.cuda)
            for ss in range(ns):
                out[ss, classpred[ss]] = 1
            return out

    def update_exemplars(self, nc, training=True):
        self.model.eval()
        with torch.no_grad():
            # Reduce exemplar set by updating value of num. exemplars per class
            self.num_exemplars = int(
                self.memory_size / (len(nc) + len(self.mem_class_x.keys()))
            )

            for c in self.mem_class_x.keys():
                self.mem_class_x[c] = self.mem_class_x[c][: self.num_exemplars]
                self.mem_class_y[c] = self.mem_class_y[c][: self.num_exemplars]

            for c in nc:
                # Find indices of examples of class c
                indxs = (self.buf_y == c).nonzero(as_tuple=False).squeeze()
                mem_x_c = move_cuda(torch.index_select(
                    self.buf_x, 0, indxs), self.cuda)

                # Compute feature vectors of examples of class c
                self.model(mem_x_c)
                memf_c = self.model.feature_vector

                # Compute the mean feature vector of class
                num_samples = torch.tensor(memf_c.size(0))
                sum_memf_c = memf_c.sum(0)
                sum_memf_c = hvd.allreduce(
                    sum_memf_c, name=f"sum_memf_{c}", op=hvd.Sum)
                sum_num_samples = hvd.allreduce(
                    num_samples, name=f"sum_num_samples_{c}", op=hvd.Sum
                )

                mean_memf_c = sum_memf_c / sum_num_samples
                mean_memf_c = mean_memf_c.view(1, self.model.num_features)

                # Compute the distance between each feature vector of the examples of class c and the mean feature vector of class c
                dist_memf_c = torch.cdist(memf_c, mean_memf_c)
                dist_memf_c = dist_memf_c.view(dist_memf_c.size(0))

                # Find the indices the self.num_exemplars features vectors closest to the mean feature vector of class c
                indices = torch.sort(dist_memf_c)[1][: self.num_exemplars]

                # Save the self.num_exemplars examples of class c with the closest feature vector to the mean feature vector of class c
                self.mem_class_x[c] = torch.index_select(
                    mem_x_c, 0, indices).cpu()

            # Recompute outputs for distillation purposes and means for inference purposes
            for cc in self.mem_class_x.keys():
                self.mem_class_y[cc] = self.model(
                    move_cuda(self.mem_class_x[cc], self.cuda)
                ).cpu()
                tmp_features = self.model.feature_vector

                num_samples = torch.tensor(tmp_features.size(0))
                sum_memf_c = tmp_features.sum(0)
                sum_memf_c = hvd.allreduce(
                    sum_memf_c, name=f"sum_memf_{c}", op=hvd.Sum)
                sum_num_samples = hvd.allreduce(
                    num_samples, name=f"sum_num_samples_{c}", op=hvd.Sum
                )

                self.mem_class_means[cc] = sum_memf_c / sum_num_samples
                self.mean_features = torch.stack(
                    tuple(self.mem_class_means.values())
                ).cpu()

        del tmp_features
        if training:
            self.buf_x = None
            self.buf_y = None
