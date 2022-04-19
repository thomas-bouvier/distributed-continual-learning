import copy
import math
import numpy as np
import logging
import random
import time
import torch
import torch.nn as nn

from agents.base import Agent
from utils.utils import get_device, move_cuda
from utils.meters import AverageMeter, accuracy


class Representative(object):
    """
    Representative sample of the algorithm
    """

    def __init__(self, x, y, net_output=None):
        """
        Creates a Representative object
        :param x: the value of the representative (i.e. the image)
        :param y: the label attached to the value x
        :param net_output: the output that the neural network gives to the sample
        """
        self.x = x
        self.y = y
        self.net_output = net_output
        self.weight = 1.0

    def __eq__(self, other):
        if isinstance(other, Representative.__class__):
            return self.x.__eq__(other.x)
        return False

    def get_size(self):
        return (self.x.element_size() * self.x.nelement()) / 1000000000


"""This implementation doesn't work because of CUDA OOM issues.
"""


class nil_list_agent(Agent):
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
        super(nil_list_agent, self).__init__(
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

        self.num_representatives = config.get(
            "num_representatives", 60
        )  # number of stored examples per class
        self.num_candidates = config.get("num_candidates", 20)
        self.batch_size = config.get("batch_size")

        self.num_reps = 0
        self.representatives = [[] for _ in range(model.num_classes)]
        # self.representatives_x = ma.masked_all((model.num_classes, self.num_representatives), torch.Tensor)
        self.class_count = [0 for _ in range(model.num_classes)]

        self.mask = torch.as_tensor(
            [0.0 for _ in range(self.model.num_classes)],
            device=torch.device(get_device(self.cuda)),
        )

    def get_samples(self):
        """Select or retrieve the representatives from the data

        :return: a list of num_candidates representatives.
        """
        # Naive version
        repr_list = [a for sublist in self.representatives for a in sublist]
        if len(repr_list) > 0:
            return random.sample(repr_list, min(self.num_candidates, len(repr_list)))
        else:
            return []

        # Numpy version
        # repr_list = self.representatives_x.compressed()
        # if len(repr_list) > 0:
        #    return np.random.choice(repr_list, (min(self.num_candidates, len(repr_list)),), replace=False)
        # else:
        #    return np.array([])

    def get_num_representatives(self):
        return sum(len(nclass) for nclass in self.representatives)

    def get_memory_size(self):
        return sum(rep.get_size() for nclass in self.representatives for rep in nclass)

    def accumulate(self, x, y):
        """Modify the representatives list by selecting candidates randomly from the
        incoming data x and the current list of representatives.

        In this version, all c first candidates of the incoming batch x are injected
        into the episodic memory.
        """
        # Naive version
        rand_indices = torch.from_numpy(np.random.permutation(len(x)))
        x = x[rand_indices]
        y = y[rand_indices]
        for i in range(min(self.num_candidates, len(x))):
            nclass = y[i].item()
            self.class_count[nclass] += 1
            if len(self.representatives[nclass]) >= self.num_representatives:
                rand = random.randrange(len(self.representatives[nclass]))
                self.representatives[nclass][rand] = Representative(x[i], y[i])
            else:
                self.representatives[nclass].append(Representative(x[i], y[i]))

        # for i in range(min(self.num_candidates, len(x))):
        #    nclass = y[i].item()
        #    self.class_count[nclass] += 1
        #    if self.class_count[nclass] > self.num_representatives:
        #        rand = random.randrange(self.num_representatives)
        #        # Doesn't work
        #        #del self.representatives_x[nclass][rand]
        #        self.representatives_x[nclass][rand] = x[i]
        #    else:
        #        self.representatives_x[nclass][self.class_count[nclass]-1] = x[i]

        self.recalculate_weights()

    def recalculate_weights(self):
        """
        Reassign the weights of the representatives
        """
        total_count = np.sum(self.class_count)
        # This version proposes that the total weight of representatives is
        # calculated from the proportion of candidates with respect to the batch.
        # E.g. a batch of 100 images and 10 are num_candidates selected, total_weight = 10
        total_weight = (self.batch_size * 1.0) / self.num_candidates
        # The total_weight is adjusted to the proportion between candidates
        # and representatives.
        total_weight *= total_count / \
            np.sum([len(cls) for cls in self.representatives])
        probs = [count / total_count for count in self.class_count]
        for i in range(len(self.representatives)):
            if self.class_count[i] > 0:
                for rep in self.representatives[i]:
                    # This version uses natural log as a stabilizer
                    rep.weight = max(
                        math.log(probs[i].item() * total_weight), 1.0)

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

        # Add the new classes to the mask
        torch.cuda.nvtx.range_push("Create mask")
        nc = set([data[1] for data in train_data_regime.get_data()])
        for y in nc:
            self.mask[y] = 1.0
        torch.cuda.nvtx.range_pop()

        self.criterion = nn.CrossEntropyLoss(
            weight=self.mask, reduction="none")

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

        for i_batch, (x, y, t) in enumerate(data_regime.get_loader()):
            torch.cuda.nvtx.range_push(f"Batch {i_batch}")

            output, loss = self._step(
                i_batch, x, y, training=training, average_output=average_output
            )

            # measure accuracy and record loss
            prec1, prec5 = accuracy(
                output[: y.size(0)],
                y,
                topk=(1, min(self.model.num_classes, 5)),
            )
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
                        self.epoch + 1,
                        i_batch,
                        len(data_regime.get_loader()),
                        phase="TRAINING" if training else "EVALUATING",
                        meters=meters,
                    )
                )

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
                        self.writer.add_scalar(
                            "num_representatives",
                            self.get_num_representatives(),
                            self.global_steps,
                        )
                        self.writer.add_scalar(
                            "memory_size",
                            self.get_memory_size(),
                            self.global_steps,
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

            if training:
                # Get the representatives
                reps = self.get_samples()
                num_reps = len(reps)

            # Create batch weights
            w = torch.ones(len(x), device=torch.device(get_device(self.cuda)))
            torch.cuda.nvtx.range_push("Copy to device")
            x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)
            torch.cuda.nvtx.range_pop()

            if training and num_reps > 0:
                torch.cuda.nvtx.range_push("Combine batches")
                rep_values = move_cuda(torch.stack(
                    [rep.x for rep in reps]), self.cuda)
                rep_labels = move_cuda(torch.stack(
                    [rep.y for rep in reps]), self.cuda)
                rep_weights = torch.as_tensor(
                    [rep.weight for rep in reps],
                    device=torch.device(get_device(self.cuda)),
                )
                # Concatenate the training samples with the representatives
                w = torch.cat((w, rep_weights))
                x = torch.cat((x, rep_values))
                y = torch.cat((y, rep_labels))
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Forward pass")
            output = self.model(x)
            if training:
                loss = self.criterion(output, y)
            else:
                loss = nn.CrossEntropyLoss()(output, y)
            torch.cuda.nvtx.range_pop()

            if training:
                # Leads to decreased accuracy
                # total_weight = hvd.allreduce(torch.sum(w), name='total_weight', op=hvd.Sum)
                dw = w / torch.sum(w)
                # Faster to provide the derivative of L wrt {l}^n than letting
                # pytorch computing it by itself
                loss.backward(dw)
                # SGD step
                torch.cuda.nvtx.range_push("Optimizer step")
                self.optimizer.step()
                torch.cuda.nvtx.range_pop()
                self.global_steps += 1
                self.steps += 1

                if num_reps == 0:
                    if self.buffer_cuda:
                        self.accumulate(x, y)
                    else:
                        self.accumulate(x.cpu(), y.cpu())
                else:
                    if self.buffer_cuda:
                        self.accumulate(x[:-num_reps], y[:-num_reps])
                    else:
                        self.accumulate(
                            x[:-num_reps].cpu(), y[:-num_reps].cpu())

            outputs.append(output.detach())
            total_loss += torch.mean(loss).item()

            torch.cuda.nvtx.range_pop()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss
