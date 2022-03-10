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


def memory_manager(train_data_regime, validate_data_regime, q, lock, lock_make,
                   lock_made, num_classes, num_candidates, num_representatives,
                   batch_size, cuda, size):
    representatives = [[] for _ in range(num_classes)]
    class_count = [0 for _ in range(num_classes)]
    reps_x, reps_y, reps_w = q.get()

    for task_id in range(0, len(train_data_regime.tasksets)):
        i_epoch = 0
        next_task = False

        train_data_regime.set_task_id(task_id)
        validate_data_regime.set_task_id(task_id)

        # Distribute the data
        train_data_regime.get_loader(True)
        validate_data_regime.get_loader(True)

        while not next_task:
            # Horovod: set epoch to sampler for shuffling
            train_data_regime.set_epoch(i_epoch)

            for i_batch, (x, y, t) in enumerate(train_data_regime.get_loader()):
                samples = torch.randperm(min(num_candidates, len(x)))
                inputs_batch = x[samples]
                target_batch = y[samples]

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
                    for rep in representatives[i]:
                        rep.weight = max(math.log(probs[i] * total_weight), 1.0)

                #Send next batch's candidates 
                repr_list = [a for sublist in representatives for a in sublist]
                while len(repr_list) < num_candidates + batch_size - len(x):
                    repr_list += [a for sublist in representatives for a in sublist]

                sampled = random.sample(repr_list, num_candidates + batch_size - len(x))
                n_reps_x = torch.stack([a.value for a in sampled])
                n_reps_y = torch.tensor([a.label.item() for a in sampled])
                n_reps_w = torch.tensor([a.weight for a in sampled])

                w = torch.ones(len(x))
                w = torch.cat([w, n_reps_w])
                x = torch.cat([x, n_reps_x])
                y = torch.cat([y, n_reps_y])

                info = q.get()
                if info == -2:
                    return
                if info == -1:
                    next_task = True
                    break

                assert i_batch == info

                lock_make.acquire()
                lock.acquire()
                reps_x[0] = move_cuda(x, cuda)
                reps_y[0] = move_cuda(y, cuda)
                reps_w[0] = move_cuda(w, cuda)
                lock.release()
                lock_made.release()

            if not next_task:
                # Horovod: set epoch to sampler for shuffling
                validate_data_regime.set_epoch(i_epoch)

                for i_batch, (x, y, t) in enumerate(validate_data_regime.get_loader()):
                    samples = torch.randperm(min(num_candidates, len(x)))
                    inputs_batch = x[samples]
                    target_batch = y[samples]

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
                        for rep in representatives[i]:
                            rep.weight = max(math.log(probs[i] * total_weight), 1.0)

                    #Send next batch's candidates 
                    repr_list = [a for sublist in representatives for a in sublist]
                    while len(repr_list) < num_candidates + batch_size - len(x):
                        repr_list += [a for sublist in representatives for a in sublist]

                    sampled = random.sample(repr_list, num_candidates + batch_size - len(x))
                    n_reps_x = torch.stack([a.value for a in sampled])
                    n_reps_y = torch.tensor([a.label.item() for a in sampled])
                    n_reps_w = torch.tensor([a.weight for a in sampled])

                    w = torch.ones(len(x))
                    w = torch.cat([w, n_reps_w])
                    x = torch.cat([x, n_reps_x])
                    y = torch.cat([y, n_reps_y])

                    info = q.get()
                    if info == -2:
                        return
                    if info == -1:
                        next_task = True
                        break

                    assert i_batch == info

                    lock_make.acquire()
                    lock.acquire()
                    reps_x[0] = move_cuda(x, cuda)
                    reps_y[0] = move_cuda(y, cuda)
                    reps_w[0] = move_cuda(w, cuda)
                    lock.release()
                    lock_made.release()

            i_epoch += 1


class nil_v4_agent(Agent):
    def __init__(self, model, config, optimizer, criterion, cuda, log_interval, state_dict=None):
        super(nil_v4_agent, self).__init__(model, config, optimizer, criterion, cuda, log_interval, state_dict)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.memory_size = config.get('num_representatives') # number of stored examples per class
        self.num_candidates = config.get('num_candidates', 20) # number of representatives used to increment batches and to update representatives
        self.batch_size = config.get('batch_size')

        self.mask = torch.as_tensor([0.0 for _ in range(self.model.num_classes)], device=torch.device(get_device(self.cuda)))

    def before_all_tasks(self, train_data_regime, validate_data_regime):
        self.x_dim = list(train_data_regime.tasksets[0][0][0].size())

        self.reps_x = move_cuda(torch.zeros([1, self.num_candidates + self.batch_size] + self.x_dim), self.cuda).share_memory_()
        self.reps_y = move_cuda(torch.zeros([1, self.num_candidates + self.batch_size], dtype=torch.long), self.cuda).share_memory_()
        self.reps_w = move_cuda(torch.zeros([1, self.num_candidates + self.batch_size]), self.cuda).share_memory_()

        self.q = mp.Queue()
        self.lock = mp.Lock()
        self.lock_make = mp.Lock()
        self.lock_made = mp.Lock()
        self.lock_made.acquire()

        self.p = mp.Process(target=memory_manager, args=[train_data_regime, validate_data_regime, self.q, self.lock, self.lock_make, self.lock_made, self.model.num_classes, self.num_candidates, self.memory_size, self.batch_size, self.cuda])
        self.p.start()
        self.q.put((self.reps_x, self.reps_y, self.reps_w))

    def after_all_tasks(self):
        self.q.put(-2)
        self.p.join()
        self.q.close()

    def before_every_task(self, task_id, train_data_regime):
        self.steps = 0

        if self.best_model is not None:
            logging.debug(f"Loading best model with minimal eval loss ({self.minimal_eval_loss})..")
            self.model.load_state_dict(self.best_model)
            self.minimal_eval_loss = float('inf')
        if task_id > 0:
            if self.config.get('reset_state_dict', False):
                logging.debug(f"Resetting model internal state..")
                self.model.load_state_dict(copy.deepcopy(self.initial_snapshot))
            self.optimizer.reset(self.model.parameters())

        # Create mask so the loss is only used for classes learnt during this task
        torch.cuda.nvtx.range_push("Create mask")
        nc = set([data[1] for data in train_data_regime.get_data()])
        for y in nc:
            mask[y] = 1.0
        torch.cuda.nvtx.range_pop()

        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

        self.x = move_cuda(torch.zeros([self.num_candidates + self.batch_size] + self.x_dim), self.cuda)
        self.y = move_cuda(torch.zeros([self.num_candidates + self.batch_size], dtype=torch.long), self.cuda)
        self.w = move_cuda(torch.zeros([self.num_candidates + self.batch_size]), self.cuda)

    def after_every_task(self):
        self.q.put(-1)

    """
    Forward pass for the current epoch
    """
    def loop(self, data_regime, average_output=False, training=False):
        prefix='train' if training else 'val'
        meters = {metric: AverageMeter(f"{prefix}_{metric}")
                  for metric in ['loss', 'prec1', 'prec5']}
        start = time.time()
        step_count = 0

        parallel_batch_size = hvd.size() * self.batch_size
        data_size = len(data_regime.get_data())
        for i_batch in range(math.ceil(data_size / parallel_batch_size)):
            torch.cuda.nvtx.range_push(f"Batch {i_batch}")
            self.q.put(i_batch)

            output, loss = self._step(i_batch, None, None, training=training,
                                      average_output=average_output)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, self.y, topk=(1, min(self.model.num_classes, 5)))
            meters['loss'].update(loss)
            meters['prec1'].update(prec1, self.x.size(0))
            meters['prec5'].update(prec5, self.x.size(0))

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
                                data=(self.x, self.y))
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

    def _step(self, i_batch, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.steps)

        #for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
        #                                         target_batch.chunk(chunk_batch, dim=0))):
        #    inputs, target = move_cuda(inputs, self.cuda), move_cuda(target, self.cuda)

        torch.cuda.nvtx.range_push(f"Wait for representatives")
        self.lock_made.acquire()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"Get representatives")
        self.lock.acquire()
        self.x.copy_(self.reps_x[0])
        self.y.copy_(self.reps_y[0])
        self.w.copy_(self.reps_w[0])
        self.lock.release()
        self.lock_make.release()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Forward pass")
        output = self.model(self.x)
        loss = self.criterion(output, self.y)
        torch.cuda.nvtx.range_pop()

        if training:
            dw = self.w / torch.sum(self.w)
            # Can be faster to provide the derivative of L wrt {l}^b than letting pytorch computing it by itself
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
