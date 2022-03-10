import horovod.torch as hvd
import math
import numpy as np
import logging
import random
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from agents.base import Agent
from utils.cl import Representative
from utils.utils import get_device, move_cuda
from utils.meters import AverageMeter, accuracy


def memory_manager(q_new_batch, reps_x, reps_y, reps_w, lock, num_classes, num_candidates, num_representatives, batch_size, cuda):
    representatives = [[] for _ in range(num_classes)]
    class_count = [0 for _ in range(num_classes)]

    while True:
        # Get new batch
        item = q_new_batch.get()
        if item == 0:
            return

        x = move_cuda(item[0].clone(), cuda)
        y = move_cuda(item[1].clone(), cuda)
        del item

        rand_indices = torch.from_numpy(np.random.permutation(len(x)))
        x = x[rand_indices]  # The data is ordered according to the indices
        y = y[rand_indices]
        for i in range(min(num_candidates, len(x))):
            nclass = y[i].item()
            class_count[nclass] += 1
            if len(selrepresentatives[nclass]) >= num_representatives:
                del representatives[nclass][num_representatives-1]
            representatives[nclass].append(Representative(x[i], y[i]))

        # Update weights of reps
        total_count = sum(class_count)
        total_weight = (batch_size * 1.0) / num_candidates
        total_weight *= (total_count / np.sum([len(cls) for cls in representatives]))
        probs = [count / total_count for count in class_count]

        for i in range(len(representatives)):
            if class_count[i] > 0:
                for rep in representatives[i]:
                    # This version uses natural log as an stabilizer
                    rep.weight = max(math.log(probs[i] * total_weight), 1.0)

        #Send next batch's candidates 
        repr_list = [a for sublist in representatives for a in sublist]
        if len(repr_list) > num_candidates:
            # Version without concurrent reads

            #samples = random.sample(repr_list, num_candidates)
            #n_reps_x = torch.stack([a.value for a in samples])
            #n_reps_y = torch.tensor([a.label.item() for a in samples])
            #n_reps_w = torch.tensor([a.weight for a in samples])
            #
            #lock.acquire()
            #reps_x -= reps_x - n_reps_x
            #reps_y -= reps_y - n_reps_y
            #reps_w -= reps_w - n_reps_w
            #lock.release()samples = random.sample(repr_list, num_candidates)

            # Version with concurrent reads

            samples = random.sample(repr_list, num_candidates)
            n_reps_x = torch.stack([a.value for a in samples]) - reps_x
            n_reps_y = torch.stack([a.label for a in samples]) - reps_y
            n_reps_w = move_cuda(torch.tensor([a.weight for a in samples]), cuda) - reps_w

            lock.acquire()
            reps_x += n_reps_x
            reps_y += n_reps_y
            reps_w += n_reps_w
            lock.release()


class nil_v2_agent(Agent):
    def __init__(self, model, config, optimizer, criterion, cuda, log_interval, state_dict=None):
        super(nil_v2_agent, self).__init__(model, config, optimizer, criterion, cuda, log_interval, state_dict)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.memory_size = config.get('num_representatives', 6000)   # number of stored examples per class
        self.num_candidates = config.get('num_candidates', 20)   # number of representatives used to increment batches and to update representatives
        self.batch_size = config.get('batch_size')

        self.mask = torch.as_tensor([0.0 for _ in range(self.model.num_classes)], device=torch.device(get_device(self.cuda)))

    def before_all_tasks(self, train_data_regime):
        x_dim = list(train_data_regime.tasksets[0][0][0].size())

        self.reps_x = move_cuda(torch.zeros([self.num_candidates] + x_dim), self.cuda).share_memory_()
        self.reps_y = move_cuda(torch.zeros([self.num_candidates], dtype=torch.long), self.cuda).share_memory_()
        self.reps_w = move_cuda(torch.zeros([self.num_candidates]), self.cuda).share_memory_()
        
        self.q_new_batch = mp.Queue()
        self.lock = mp.Lock()

        self.p = mp.Process(target=memory_manager, args=[self.q_new_batch, self.reps_x, self.reps_y, self.reps_w, self.lock, self.model.num_classes, self.num_candidates, self.memory_size, self.batch_size, self.cuda])
        self.p.start()

    def after_all_tasks(self):
        self.q_new_batch.put(0)
        self.p.join()
        self.q_new_batch.close()

    def before_every_task(self, task_id, train_data_regime):
        self.steps = 0

        # Distribute the data
        torch.cuda.nvtx.range_push("Distribute dataset")
        train_data_regime.get_loader(True)
        torch.cuda.nvtx.range_pop()

        if self.best_model is not None:
            logging.debug(f"Loading best model with minimal eval loss ({self.minimal_eval_loss})..")
            self.model.load_state_dict(self.best_model)
            self.minimal_eval_loss = float('inf')
        if task_id > 0:
            if self.config.get('reset_state_dict', False):
                logging.debug(f"Resetting model internal state..")
                self.model.load_state_dict(copy.deepcopy(self.initial_snapshot))
            self.optimizer.reset(self.model.parameters())

        # Add the new classes to the mask
        torch.cuda.nvtx.range_push("Create mask")
        nc = set([data[1] for data in train_data_regime.get_data()])
        for y in nc:
            self.mask[y] = 1.0
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
            meters['loss'].update(loss)
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
                    self.writer.add_scalar(f"{prefix}_loss", meters['loss'].avg, self.global_steps)
                    self.writer.add_scalar(f"{prefix}_prec1", meters['prec1'].avg, self.global_steps)
                    self.writer.add_scalar(f"{prefix}_prec5", meters['prec5'].avg, self.global_steps)
                    if training:
                        self.writer.add_scalar('lr', self.optimizer.get_lr()[0], self.global_steps)
                    self.writer.flush()
                if self.watcher is not None:
                    self.observe(trainer=self,
                                model=self.model,
                                optimizer=self.optimizer,
                                data=(x, y))
                    self.stream_meters(meters, prefix=prefix)
                    if training:
                        self.write_stream('lr',
                                         (self.global_steps, self.optimizer.get_lr()[0]))
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
            self.optimizer.update(self.epoch, self.steps)

        for i, (x, y) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                       target_batch.chunk(chunk_batch, dim=0))):
            torch.cuda.nvtx.range_push(f"Chunk {i}")

            if training:
                torch.cuda.nvtx.range_push(f"Get representatives")
                self.lock.acquire()
                rep_values = self.reps_x.clone()
                rep_labels = self.reps_y.clone()
                rep_weights = self.reps_w.clone()
                self.lock.release()
                if rep_weights[-1] == 0 :
                    n_reps = 0
                else:
                    n_reps = len(self.reps_x)
                torch.cuda.nvtx.range_pop()

                # select next samples to become representatives
                torch.cuda.nvtx.range_push(f"Update representatives")
                rand_indices = torch.from_numpy(np.random.permutation(min(self.num_candidates, len(x))))
                reps_x = x[rand_indices].clone().share_memory_()  # The data is ordered according to the indices
                reps_y = y[rand_indices].clone().share_memory_()
                self.q_new_batch.put((reps_x, reps_y))
                torch.cuda.nvtx.range_pop()

            # Create batch weights
            w = torch.ones(len(x))
            torch.cuda.nvtx.range_push("Copy to device")
            x, y, w = move_cuda(x, self.cuda), move_cuda(y, self.cuda), move_cuda(w, self.cuda)
            torch.cuda.nvtx.range_pop()

            if training and n_reps > 0:
                # Concatenates the training samples with the representatives
                torch.cuda.nvtx.range_push("Combine batches")
                w = torch.cat((w, rep_weights))
                x = torch.cat((x, rep_values))
                y = torch.cat((y, rep_labels))
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Forward pass")
            output = self.model(x)
            loss = self.criterion(output, y)
            torch.cuda.nvtx.range_pop()

            if training:
                dw = w / torch.sum(w)
                # Faster to provide the derivative of L wrt {l}^b than letting pytorch computing it by itself
                loss.backward(dw)
                # SGD step
                torch.cuda.nvtx.range_push("Optimizer step")
                self.optimizer.step()
                torch.cuda.nvtx.range_pop()
                self.global_steps += 1
                self.steps += 1
        
            outputs.append(output.detach())
            total_loss += torch.mean(loss)

            torch.cuda.nvtx.range_pop()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss
