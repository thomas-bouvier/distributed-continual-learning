import horovod.torch as hvd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from continuum.tasks import split_train_val

from agents.base import Agent
from utils.utils import move_cuda
from meters import AverageMeter, accuracy


def memory_manager(dataset, q, lock, lock_make, lock_made, num_classes, num_candidates, num_representatives, batch_size, size, rank):
    representatives = [[] for _ in range(num_classes)]
    class_count = [0 for _ in range(num_classes)]
    buffer_sizeed_reps = []

    reps_x, reps_y, reps_w = q.get()
    device = torch.device(f'cuda:{rank % 4}')

    for task_id, taskset in enumerate(dataset):
        # Partition the training data between the workers
        train_sampler, train_loader = distribute(taskset, batch_size, size=size, rank=rank)
        epoch = 0
        next_task = False

        while not next_task:
            # Horovod: set epoch to sampler for shuffling.
            train_sampler.set_epoch(epoch)

            for batch_id, (x, y, _) in enumerate(train_loader):
                samples = torch.randperm(min(num_candidates, len(x)))

                image_batch = x[samples]
                target_batch = y[samples]

                for i in range(len(image_batch)):
                    buffer_sizeed_reps.append(Representative(image_batch[i].clone(), target_batch[i].clone()))

                for i, _ in enumerate(buffer_sizeed_reps):
                    nclass = int(buffer_sizeed_reps[i].label.item())
                    representatives[nclass].append(buffer_sizeed_reps[i])
                    class_count[nclass] += 1

                buffer_sizeed_reps = []

                for i in range(len(representatives)):
                    rand_indices = np.random.permutation(len(representatives[i]))
                    representatives[i] = [representatives[i][j] for j in rand_indices]
                    representatives[i] = representatives[i][:nb_representatives]

                # Update weights of reps
                total_count = sum(class_count)
                total_weight = (batch_size * 1.0) / n_candidates
                total_weight *= (total_count / np.sum([len(cls) for cls in representatives]))

                probs = [count / total_count for count in class_count]

                for i in range(len(representatives)):
                    for rep in representatives[i]:
                        rep.weight = max(math.log(probs[i] * total_weight), 1.0)

                #Send next batch's candidates 
                repr_list = [a for sublist in representatives for a in sublist]

                while len(repr_list) < n_candidates + batch_size - len(x):
                    repr_list += [a for sublist in representatives for a in sublist]

                sampled = random.sample(repr_list,  n_candidates + batch_size - len(x))

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

                assert info == batch_id

                lock_make.acquire()

                lock.acquire()
                reps_x[0] = x.to(device)
                reps_y[0] = y.to(device)
                reps_w[0] = w.to(device)
                lock.release()

                lock_made.release()

            epoch +=1


class nil_v4_agent(Agent):
    def __init__(self, model, config, optimizer, criterion, cuda, log_interval, state_dict=None):
        super(nil_v4_agent, self).__init__(model, config, optimizer, criterion, cuda, log_interval, state_dict)

        self.class_count = [0 for _ in range(model.num_classes)]

        self.memory_size = config.get('num_representatives') # number of stored examples per class
        self.num_candidates = config.get('num_candidates', 20) # number of representatives used to increment batches and to update representatives
        self.batch_size = config.get('batch_size')
        self.epochs = config.get('max_epochs') # Maximum number of epochs
        self.is_early_stop = config.get('is_early_stop')

        self.mask = torch.as_tensor([False for _ in range(model.num_classes)])
        self.buffer_size = 1 # frequency of representatives updtate (not used)
        self.val_set = None

    def before_all_tasks(self, taskets):
        self.x_dim = list(taskets[0][0][0].size())

        self.reps_x = move_cuda(torch.zeros([1, self.num_candidates + self.batch_size] + self.x_dim), self.cuda).share_memory_()
        self.reps_y = move_cuda(torch.zeros([1, self.num_candidates + self.batch_size], dtype=torch.long), self.cuda).share_memory_()
        self.reps_w = move_cuda(torch.zeros([1, self.num_candidates + self.batch_size]), self.cuda).share_memory_()

        self.q = mp.Queue()
        self.lock = mp.Lock()
        self.lock_make = mp.Lock()
        self.lock_made = mp.Lock()
        self.lock_made.acquire()

        self.p = mp.Process(target=memory_manager, args=[taskets, self.q, self.lock, self.lock_make, self.lock_made, self.model.num_classes, self.num_candidates, self.memory_size, self.batch_size, hvd.size(), hvd.rank()])
        self.p.start()
        self.q.put((self.reps_x, self.reps_y, self.reps_w))

    def after_all_tasks(self):
        self.q.put(-2)
        self.p.join()
        self.q.close()

    def before_every_task(self, task_id, train_taskset):
        # Create mask so the loss is only used for classes learnt during this task
        nc = set([data[1] for data in train_taskset])
        mask = torch.tensor([False for _ in range(self.model.num_classes)])
        for y in nc:
            mask[y] = True
        self.mask = move_cuda(mask.float(), self.cuda)

        self.x = move_cuda(torch.zeros([self.num_candidates + self.batch_size] + self.x_dim), self.cuda)
        self.y = move_cuda(torch.zeros([self.num_candidates + self.batch_size], dtype=torch.long), self.cuda)
        self.w = move_cuda(torch.zeros([self.num_candidates + self.batch_size]), self.cuda)

    def after_every_task(self):
        self.q.put(-1)

    """
    Forward pass for the current epoch
    """
    def loop(self, data_loader, average_output=False, training=False):
        meters = {metric: AverageMeter()
                  for metric in ['loss', 'prec1', 'prec5']}

        for i_batch in range(int(len(data_loader) / (hvd.size() * self.batch_size)) + ((len(data_loader) % (hvd.size() * self.batch_size)) != 0)):
            self.q.put(i_batch)

            self.lock_made.acquire()
            self.lock.acquire()
            self.x.copy_(self.reps_x[0])
            self.y.copy_(self.reps_y[0])
            self.w.copy_(self.reps_w[0])
            self.lock.release()
            self.lock_make.release()

            output, loss = self._step(i_batch,
                            self.x,
                            self.y,
                            training=training,
                            average_output=average_output)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))

            mlflow.log_metrics({
                'loss': float(loss),
                'prec1': float(prec1),
                'prec5': float(prec5)
            }, step=self.epoch)

            if i_batch % self.log_interval == 0 or i_batch == len(data_loader) - 1:
                print('{phase}: epoch: {0} [{1}/{2}]\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             .format(
                                 self.epoch+1, i_batch, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))

        meters = {name: meter.avg for name, meter in meters.items()}
        meters['error1'] = 100. - meters['prec1']
        meters['error5'] = 100. - meters['prec5']

        return meters

    def _step(self, i_batch, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            inputs, target = move_cuda(inputs, self.cuda), move_cuda(target, self.cuda)

            output = self.model(inputs)
            loss = self.criterion(output, target)
            dw = w / torch.sum(w)

            if training:
                # Can be faster to provide the derivative of L wrt {l}^b than letting pytorch computing it by itself
                loss.backward(dw)
                # SGD step
                self.optimizer.step()
                self.training_steps += 1

            outputs.append(output.detach())
            total_loss += float(loss)

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss


class Representative(object):
    """
    Representative sample of the algorithm
    """
    def __init__(self, value, label, net_output=None):
        """
        Creates a Representative object
        :param value: the value of the representative (i.e. the image)
        :param metric: the value of the metric
        :param iteration: the iteration at which the sample was selected as representative
        :param megabatch: the current megabatch
        :param net_output: the output that the neural network gives to the sample
        """
        self.value = value
        self.label = label
        self.net_output = net_output
        self.weight = 1.0


    def __eq__(self, other):
        if isinstance(other, Representative.__class__):
            return self.value.__eq__(other.value)
        return False
