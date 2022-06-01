import copy
import horovod.torch as hvd
import numpy as np
import logging
import time
import torch
import torch.nn as nn
import wandb

from agents.base import Agent
from utils.utils import move_cuda
from utils.meters import AverageMeter, accuracy


class icarl_agent(Agent):
    def __init__(
        self,
        model,
        config,
        optimizer,
        criterion,
        cuda,
        buffer_cuda,
        log_interval,
        state_dict=None,
    ):
        super(icarl_agent, self).__init__(
            model,
            config,
            optimizer,
            criterion,
            cuda,
            buffer_cuda,
            log_interval,
            state_dict,
        )

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        # Modified parameters
        self.num_exemplars = 0
        self.memory_size = config.get(
            "num_representatives", 6000) * model.num_classes
        self.num_candidates = config.get("num_candidates", 20)

        # Store current batch on CPU for current epoch
        self.buf_x = None
        self.buf_y = None
        # Store exemplars by class
        self.mem_class_x = {}
        self.mem_class_y = {}
        self.mem_class_means = {}

        # Setup distillation losses
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.lsm = nn.LogSoftmax(dim=1)
        self.sm = nn.Softmax(dim=1)

    def before_every_task(self, task_id, train_data_regime):
        self.steps = 0

        # Distribute the data
        torch.cuda.nvtx.range_push("Distribute dataset")
        train_data_regime.get_loader(True)
        torch.cuda.nvtx.range_pop()

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
            self.optimizer.reset(self.model.parameters())

        # Create mask so the loss is only used for classes learnt during this task
        torch.cuda.nvtx.range_push("Create mask")
        self.nc = set([data[1] for data in train_data_regime.get_data()])
        mask = torch.tensor([False for _ in range(self.model.num_classes)])
        for y in self.nc:
            mask[y] = True
        self.mask = move_cuda(mask.float(), self.cuda)
        torch.cuda.nvtx.range_pop()

        self.criterion = nn.CrossEntropyLoss(weight=self.mask)

        # Distillation
        self.should_distill = self.mem_class_x != {}
        if self.should_distill:
            # Updated before each task
            self.mem_x = torch.cat(
                [samples.clone() for samples in self.mem_class_x.values()]
            )
            self.mem_y = torch.cat(
                [targets.clone() for targets in self.mem_class_y.values()]
            )

    def after_every_task(self):
        torch.cuda.nvtx.range_push("Update exemplars")
        self.update_exemplars(self.nc)
        torch.cuda.nvtx.range_pop()

    """
    Forward pass for the current epoch
    """

    def loop(self, data_regime, average_output=False, training=False):
        prefix = "train" if training else "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5"]
        }
        start = time.time()
        step_count = 0

        # Distillation, increment taskset
        cand_x, cand_y = None, None
        if self.should_distill and training:
            torch.cuda.nvtx.range_push("Increment taskset")
            cand_x, cand_y = self.make_candidates(
                len(data_regime.get_loader()), self.mem_x, self.mem_y
            )
            torch.cuda.nvtx.range_pop()

        for i_batch, (x, y, t) in enumerate(data_regime.get_loader()):
            torch.cuda.nvtx.range_push(f"Batch {i_batch}")

            output, loss = self._step(
                i_batch,
                x,
                y,
                cand_x,
                cand_y,
                training=training,
                average_output=average_output,
            )

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output[: y.size(0)], y, topk=(1, 5))
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

                if hvd.rank() == 0 and hvd.local_rank() == 0:
                    wandb.log({f"{prefix}_loss": meters["loss"].avg,
                        "step": self.global_steps,
                        "epoch": self.global_epoch,
                        "batch": i_batch,
                        f"{prefix}_prec1": meters["prec1"].avg,
                        f"{prefix}_prec5": meters["prec5"].avg})
                    if training:
                        wandb.log({"lr": self.optimizer.get_lr()[0],
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
                            "lr", self.optimizer.get_lr()[0], self.global_steps
                        )
                    self.writer.flush()
                if self.watcher is not None:
                    self.observe(
                        trainer=self,
                        model=self.model,
                        optimizer=self.optimizer,
                        data=(x, y),
                    )
                    self.stream_meters(meters, prefix=prefix)
                    if training:
                        self.write_stream(
                            "lr",
                            (self.global_steps, self.optimizer.get_lr()[0]),
                        )
            torch.cuda.nvtx.range_pop()
            step_count += 1
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
        inputs_batch,
        target_batch,
        cand_x,
        cand_y,
        training=False,
        average_output=False,
        chunk_batch=1,
    ):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.steps)

        for i, (x, y) in enumerate(
            zip(
                inputs_batch.chunk(chunk_batch, dim=0),
                target_batch.chunk(chunk_batch, dim=0),
            )
        ):
            torch.cuda.nvtx.range_push(f"Chunk {i}")

            torch.cuda.nvtx.range_push("Copy to device")
            x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)
            torch.cuda.nvtx.range_pop()

            if training:
                torch.cuda.nvtx.range_push("Store batch")
                if self.buf_x is None:
                    self.buf_x = x.detach().cpu()
                    self.buf_y = y.detach().cpu()
                else:
                    self.buf_x = torch.cat((self.buf_x, x.detach().cpu()))
                    self.buf_y = torch.cat((self.buf_y, y.detach().cpu()))
                torch.cuda.nvtx.range_pop()

                if self.should_distill:
                    x = torch.cat((x, cand_x[i_batch]))

                torch.cuda.nvtx.range_push("Forward pass")
                output = self.model(x)
                loss = self.criterion(output[: y.size(0)], y)
                torch.cuda.nvtx.range_pop()

                # Compute distillation loss
                if self.should_distill:
                    torch.cuda.nvtx.range_push("Compute distillation loss")
                    loss += self.kl(
                        self.lsm(output[y.size(0):]), self.sm(cand_y[i_batch])
                    )
                    torch.cuda.nvtx.range_pop()

                # accumulate gradient
                loss.backward()
                # SGD step
                torch.cuda.nvtx.range_push("Optimizer step")
                self.optimizer.step()
                torch.cuda.nvtx.range_pop()
                self.global_steps += 1
                self.steps += 1
            else:
                torch.cuda.nvtx.range_push("Forward pass")
                output = self.forward(x)
                loss = self.criterion(output[: y.size(0)], y)
                torch.cuda.nvtx.range_pop()

            outputs.append(output.detach())
            total_loss += loss

            torch.cuda.nvtx.range_pop()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss

    def after_every_epoch(self):
        if self.epoch + 1 != self.epochs:
            self.buf_x = None
            self.buf_y = None

    def make_candidates(self, num_batches, mem_x, mem_y):
        """Shuffle and move the minibatch to CUDA memory.

        Args:
            num_batches (str): the number of batches
            mem_x (tensor): x batches
            mem_y (tensor): y batches

        Returns:
            The list of the python-grid5000 jobs retrieved.
        """
        if self.mem_class_x == {}:
            return None, None

        num_mem = mem_x.size(0)
        perm = np.random.permutation(num_mem)
        while len(perm) < self.num_candidates * num_batches:
            perm = np.concatenate([perm, np.random.permutation(num_mem)])
        perms = []
        for i in range(num_batches):
            perms.append(perm[: self.num_candidates])
            perm = perm[self.num_candidates:]
        cand_x = torch.stack(
            [torch.stack([mem_x[index] for index in perm]) for perm in perms]
        )
        cand_y = torch.stack(
            [torch.stack([mem_y[index] for index in perm]) for perm in perms]
        )

        return move_cuda(cand_x, self.cuda), move_cuda(cand_y, self.cuda)

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

            out = move_cuda(torch.zeros(ns, self.model.num_classes), self.cuda)
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


def icarl(model, config, optimizer, criterion, cuda, buffer_cuda, log_interval):
    return icarl_agent(
        model, config, optimizer, criterion, cuda, buffer_cuda, log_interval
    )
