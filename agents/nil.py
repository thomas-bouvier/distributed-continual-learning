import horovod.torch as hvd
import math
import numpy as np
import logging
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base import Agent
from agents.nil_v1 import nil_v1_agent
from agents.nil_v2 import nil_v2_agent
from agents.nil_v3 import nil_v3_agent
from agents.nil_v4 import nil_v4_agent
from utils.cl import Representative
from utils.utils import move_cuda
from utils.meters import AverageMeter, accuracy

class nil_agent(Agent):
    def __init__(self, model, config, optimizer, criterion, cuda, log_interval, state_dict=None):
        super(nil_agent, self).__init__(model, config, optimizer, criterion, cuda, log_interval, state_dict)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.representatives = [[] for _ in range(model.num_classes)]
        self.class_count = [0 for _ in range(model.num_classes)]
        self.buffer_sizeed_reps = []

        self.memory_size = config.get('num_representatives', 6000)   # number of stored examples per class
        self.num_candidates = config.get('num_candidates', 20)   # number of representatives used to increment batches and to update representatives
        self.batch_size = config.get('batch_size')
        self.epochs = config.get('epochs') # Maximum number of epochs

        mask = torch.as_tensor([False for _ in range(self.model.num_classes)])
        self.mask = move_cuda(mask.float(), self.cuda)

    def get_representatives(self):
        """
        Selects or retrieves the representatives from the data

        :return: a list of num_candidates representatives.
        """
        repr_list = [a for sublist in self.representatives for a in sublist]
        if len(repr_list) > 0:
            samples = random.sample(repr_list, min(self.num_candidates, len(repr_list)))
            return samples
        else:
            return []

    def random_buffer(self, image_batch, target_batch, outputs):
        """
        Creates a bufferbased in random sampling

        :param image_batch: the list of images of a batch
        :param target_batch: the list of one hot labels of a batch
        :param outputs: output probabilities of the neural network
        :param iteration: current iteration of training
        :param megabatch: current megabatch
        :return: None
        """
        rand_indices = torch.from_numpy(np.random.permutation(len(outputs)))
        image_batch = image_batch.clone()[rand_indices]  # The data is ordered according to the indices
        target_batch = target_batch.clone()[rand_indices]
        for i in range(min(self.num_candidates, len(image_batch))):
            self.buffer_sizeed_reps.append(Representative(image_batch[i], target_batch[i]))

    def random_modify_representatives(self, candidate_representatives):
        """
            Modifies the representatives list according to the new data by selecting representatives randomly from the
            buffer_size and the current list of representatives

            param candidate_representatives: the num_candidates representatives from the buffer_size
            :return: None
        """
        for i, _ in enumerate(candidate_representatives):
            nclass = int(candidate_representatives[i].label.item())
            self.representatives[nclass].append(candidate_representatives[i])
            self.class_count[nclass] += 1

        for i in range(len(self.representatives)):
            rand_indices = np.random.permutation(len(self.representatives[i]))
            self.representatives[i] = [self.representatives[i][j] for j in rand_indices]
            self.representatives[i] = self.representatives[i][:self.memory_size]

        if self.memory_size > 0:
            self.recalculate_weights(self.representatives)

    def clear_buffer_size(self):
        """
        Clears the buffer_size
        :return: None
        """
        self.buffer_sizeed_reps = []

    def recalculate_weights(self, representatives):
        """
        Reassigns the weights of the representatives
        :param representatives: a list of representatives
        :return: None
        """
        total_count = np.sum(self.class_count)
        # This version proposes that the total weight of representatives is calculated from the proportion of candidate
        # representatives respect to the batch. E.g. a batch of 100 images and 10 are num_candidatesected, total_weight = 10
        total_weight = (self.batch_size * 1.0) / self.num_candidates
        # The total_weight is adjusted to the proportion between candidate representatives and actual representatives
        total_weight *= (total_count / np.sum([len(cls) for cls in representatives]))
        probs = [count / total_count for count in self.class_count]
        for i in range(len(representatives)):
            if self.class_count[i] > 0:
                weight = max(math.log(probs[i].item() * total_weight), 1.0)
                for rep in representatives[i]:
                    # This version uses natural log as an stabilizer
                    rep.weight = weight

    def before_every_task(self, task_id, train_data_regime):
        # Distribute the data
        torch.cuda.nvtx.range_push("Distribute dataset")
        train_data_regime.get_loader(True)
        torch.cuda.nvtx.range_pop()

        if self.best_model is not None:
            logging.debug(f"(not) Loading best model with minimal eval loss ({self.minimal_eval_loss})..")
            #self.model.load_state_dict(self.best_model)

        # Add the new classes to the mask
        torch.cuda.nvtx.range_push("Create mask")
        nc = set([data[1] for data in train_data_regime.get_data()])
        for y in nc:
            self.mask[y] = True
        torch.cuda.nvtx.range_pop()

        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

    """
    Forward pass for the current epoch
    """
    def loop(self, data_regime, average_output=False, training=False):
        prefix='train' if training else 'val'
        meters = {metric: AverageMeter(f"{prefix}_{metric}")
                  for metric in ['loss', 'prec1', 'prec5']}
        start = time.time()
        step_count = 0

        for i_batch, (x, y, t) in enumerate(data_regime.get_loader()):
            torch.cuda.nvtx.range_push(f"Batch {i_batch}")

            output, loss = self._step(i_batch, x, y, training=training,
                                      average_output=average_output)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output[:y.size(0)], y, topk=(1, min(self.model.num_classes, 5)))
            meters['loss'].update(loss, x.size(0))
            meters['prec1'].update(prec1, x.size(0))
            meters['prec5'].update(prec5, x.size(0))

            if i_batch % self.log_interval == 0 or i_batch == len(data_regime.get_loader()):
                logging.info('{phase}: epoch: {0} [{1}/{2}]\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             .format(
                                 self.epoch+1, i_batch, len(data_regime.get_loader()),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))

                if self.writer is not None:
                    self.writer.add_scalar(f"{prefix}_loss", meters['loss'].avg, self.training_steps)
                    self.writer.add_scalar(f"{prefix}_prec1", meters['prec1'].avg, self.training_steps)
                    if training:
                        self.writer.add_scalar('lr', self.optimizer.get_lr()[0], self.training_steps)
                    self.writer.flush()
                if self.watcher is not None:
                    self.observe(trainer=self,
                                model=self.model,
                                optimizer=self.optimizer,
                                data=(x, y))
                    self.stream_meters(meters, prefix=prefix)
                    if training:
                        self.write_stream('lr',
                                         (self.training_steps, self.optimizer.get_lr()[0]))
            torch.cuda.nvtx.range_pop()
            step_count += 1
        end = time.time()

        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters['error1'] = 100. - meters['prec1']
        meters['error5'] = 100. - meters['prec5']
        meters['time'] = end - start
        meters['step_count'] = step_count

        return meters

    def _step(self, i_batch, inputs_batch, target_batch, training=False,
              average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (x, y) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                       target_batch.chunk(chunk_batch, dim=0))):
            torch.cuda.nvtx.range_push(f"Chunk {i}")

            if training:
                # Gets the representatives
                reps = self.get_representatives()
                n_reps = len(reps)

            # Create batch weights
            w = torch.ones(len(x))
            torch.cuda.nvtx.range_push("Copy to device")
            x, y, w = move_cuda(x, self.cuda), move_cuda(y, self.cuda), move_cuda(w, self.cuda)
            torch.cuda.nvtx.range_pop()

            if training and n_reps > 0:
                torch.cuda.nvtx.range_push("Combine batches")
                rep_weights = torch.as_tensor([rep.weight for rep in reps])
                rep_weights = move_cuda(rep_weights, self.cuda)
                #hprint([rep.value for rep in reps])
                rep_values  = torch.stack([rep.value for rep in reps])
                rep_labels  = torch.stack([rep.label for rep in reps])
                # Concatenates the training samples with the representatives
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
                total_weight = hvd.allreduce(torch.sum(w), name='total_weight', op=hvd.Sum)
                dw = w / total_weight
                # Faster to provide the derivative of L wrt {l}^n than letting pytorch computing it by itself
                loss.backward(dw)
                # SGD step
                torch.cuda.nvtx.range_push("Optimizer step")
                self.optimizer.step()
                torch.cuda.nvtx.range_pop()
                self.training_steps += 1

                # Modifies the list of representatives
                if n_reps == 0:
                    self.random_buffer(x, y, output)
                else:
                    self.random_buffer(x  [:-n_reps], y[:-n_reps], output[:-n_reps])
                self.random_modify_representatives(self.buffer_sizeed_reps)
                self.clear_buffer_size()

            outputs.append(output.detach())
            total_loss += torch.mean(loss)

            torch.cuda.nvtx.range_pop()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss


def nil(model, config, optimizer, criterion, cuda, log_interval):
    implementation = config.get('implementation', '')
    agent = nil_agent
    if implementation == 'v1':
        agent = nil_v1_agent
    elif implementation == 'v2':
        agent = nil_v2_agent
    elif implementation == 'v3':
        agent = nil_v3_agent
    elif implementation == 'v4':
        agent = nil_v4_agent
    return agent(model, config, optimizer, criterion, cuda, log_interval)
