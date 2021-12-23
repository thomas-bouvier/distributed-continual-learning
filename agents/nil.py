import horovod.torch as hvd
import math
import mlflow
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from continuum.tasks import split_train_val

from agents.base import Agent
from agents.nil_v1 import nil_v1_agent
from agents.nil_v2 import nil_v2_agent
from agents.nil_v3 import nil_v3_agent
from agents.nil_v4 import nil_v4_agent
from utils.utils import move_cuda
from meters import AverageMeter, accuracy

class nil_agent(Agent):
    def __init__(self, model, config, optimizer, criterion, cuda, log_interval, state_dict=None):
        super(nil_agent, self).__init__(model, config, optimizer, criterion, cuda, log_interval, state_dict)

        if state_dict != None:
            self.model.load_state_dict(state_dict)

        self.representatives = [[] for _ in range(model.num_classes)]
        self.class_count = [0 for _ in range(model.num_classes)]
        self.buffer_sizeed_reps = []
        self.buffer_size = 1     # frequency of representatives updtate (not used)

        self.memory_size = config.get('num_representatives', 6000)   # number of stored examples per class
        self.num_candidates = config.get('num_candidates', 20)   # number of representatives used to increment batches and to update representatives
        self.batch_size = config.get('batch_size')
        self.epochs = config.get('epochs') # Maximum number of epochs

        self.val_set = None

    def __get_representatives(self):
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

    def __random_buffer(self, image_batch, target_batch, outputs):
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

    def __random_modify_representatives(self, candidate_representatives):
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
            self.__recalculate_weights(self.representatives)

    def __clear_buffer_size(self):
        """
        Clears the buffer_size
        :return: None
        """
        self.buffer_sizeed_reps = []

    def __recalculate_weights(self, representatives):
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

    def before_every_task(self, task_id, train_data_regime, validate_data_regime):
        # Distribute the data
        train_data_regime.get_loader(True)
        validate_data_regime.get_loader(True)

        # Add the new classes to the mask
        nc = set([data[1] for data in train_data_regime.get_data()])
        mask = torch.as_tensor([False for _ in range(self.model.num_classes)])
        for y in nc:
            mask[y] = True
        self.mask = move_cuda(mask.float(), self.cuda)

        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

    """
    Forward pass for the current epoch
    """
    def loop(self, data_regime, average_output=False, training=False):
        meters = {metric: AverageMeter()
                  for metric in ['loss', 'prec1', 'prec5']}

        for i_batch, item in enumerate(data_regime.get_loader()):
            inputs = item[0] # x
            target = item[1] # y

            output, loss = self._step(i_batch,
                                      inputs,
                                      target,
                                      training=training,
                                      average_output=average_output)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output[:target.size(0)], target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))

            mlflow.log_metrics({
                'loss': float(loss),
                'prec1': float(prec1),
                'prec5': float(prec5)
            }, step=self.epoch)

            if i_batch % self.log_interval == 0 or i_batch == len(data_regime.get_loader()):
                print('{phase}: epoch: {0} [{1}/{2}]\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             .format(
                                 self.epoch+1, i_batch, len(data_regime.get_loader()),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))

        meters = {name: meter.avg for name, meter in meters.items()}
        meters['error1'] = 100. - meters['prec1']
        meters['error5'] = 100. - meters['prec5']

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
            # Gets the representatives
            reps = self.__get_representatives()
            n_reps = len(reps)

            # Create batch weights
            w = torch.ones(len(x))
            x, y, w = move_cuda(x, self.cuda), move_cuda(y, self.cuda), move_cuda(w, self.cuda)

            if n_reps > 0:
                rep_weights = torch.as_tensor([rep.weight for rep in reps])
                rep_weights = move_cuda(rep_weights, self.cuda)
                #hprint([rep.value for rep in reps])
                rep_values  = torch.stack([rep.value for rep in reps])
                rep_labels  = torch.stack([rep.label for rep in reps])
                # Concatenates the training samples with the representatives
                w = torch.cat((w, rep_weights))
                x = torch.cat((x, rep_values))
                y = torch.cat((y, rep_labels))

            output = self.model(x)
            loss = self.criterion(output, y)
            total_weight = hvd.allreduce(torch.sum(w), name='total_weight', op=hvd.Sum)
            dw = w / total_weight

            if training:
                # Faster to provide the derivative of L wrt {l}^n than letting pytorch computing it by itself
                loss.backward(dw)
                # SGD step
                self.optimizer.step()
                self.training_steps += 1

            # Modifies the list of representatives
            if n_reps == 0:
                self.__random_buffer(x, y, output)
            else:
                self.__random_buffer(x[:-n_reps], y[:-n_reps], output[:-n_reps])
            if True:#total_it % self.buffer_size == 0:
                self.__random_modify_representatives(self.buffer_sizeed_reps)
                self.__clear_buffer_size()

            outputs.append(output.detach())
            total_loss += float(torch.mean(loss))

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


def nil(model, config, optimizer, criterion, cuda, log_interval):
    implementation = config.get('implementation', '')
    if implementation == 'v1':
        return nil_v1_agent(model, config, optimizer, criterion, cuda, log_interval)
    elif implementation == 'v2':
        return nil_v2_agent(model, config, optimizer, criterion, cuda, log_interval)
    elif implementation == 'v3':
        return nil_v3_agent(model, config, optimizer, criterion, cuda, log_interval)
    elif implementation == 'v4':
        return nil_v4_agent(model, config, optimizer, criterion, cuda, log_interval)
    return nil_agent(model, config, optimizer, criterion, cuda, log_interval)
