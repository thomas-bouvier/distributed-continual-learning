from meters import AverageMeter, accuracy

import mlflow
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from continuum.tasks import split_train_val

from agents.base import Agent
from utils import *
from models import *


def make_candidates(n, mem_x, mem_y, cand_x, cand_y, lock_make, lock_made, num_batches):
    for i in range(num_batches):
        lock_make.acquire()

        selection = torch.randperm(len(mem_x))[n].clone()

        nx = mem_x[selection].clone() - cand_x
        ny = mem_y[selection].clone() - cand_y

        cand_x += nx
        cand_y += ny

        lock_made.release()


class icarl_agent(Agent):
    # Re-implementation of
    # S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert.
    # iCaRL: Incremental classifier and representation learning.
    # CVPR, 2017.
    def __init__(self, model, config, optimizer, criterion, cuda, log_interval, state_dict=None):
        super(icarl_agent, self).__init__(model, config, optimizer, criterion, cuda, log_interval, state_dict)

        # Modified parameters
        self.num_exemplars = 0
        self.num_memories = config.get('num_representatives') * config.get('num_classes')
        self.num_features = config.get('num_features')
        self.num_classes = config.get('num_classes')
        self.num_candidates = config.get('num_candidates')

        # memory
        self.mem_x = None  # stores raw inputs, PxD
        self.mem_y = None
        self.mem_class_x = {}  # stores exemplars class by class
        self.mem_class_y = {} 
        self.mem_class_means = {}

        self.val_set = None

        # setup distillation losses
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.lsm = nn.LogSoftmax(dim=1)
        self.sm = nn.Softmax(dim=1)


    def before_every_task(self, task_id, train_taskset):
        # Create mask so the loss is only used for classes learnt during this task
        self.nc = set([data[1] for data in train_taskset])
        mask = torch.tensor([False for _ in range(self.num_classes)])
        for y in self.nc:
            mask[y] = True

        if self.cuda:
            self.mask = mask.float().cuda()
        else:
            self.mask = mask

        # Distillation
        if self.mem_class_x != {}:
            self.mem_x = torch.cat([samples.cpu() for samples in self.mem_class_x.values()]).share_memory_()
            self.mem_y = torch.cat([targets.cpu() for targets in self.mem_class_y.values()]).share_memory_()


    def after_every_task(self):
        self.update_examplars(self.nc)


    def _step(self, i_batch, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        if self.epoch + 1 == self.num_epochs:
            if self.mem_x is None:
                self.mem_x = inputs.detach()
                self.mem_y = target.detach()
            else:
                self.mem_x = torch.cat(self.mem_x, inputs.detach())
                self.mem_y = torch.cat(self.mem_y, target.detach())

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            if self.cuda:
                inputs, target = inputs.cuda(), target.cuda()

            # Distillation
            if self.mem_class_x != {}:
                self.lock_made.acquire()
                inputs = torch.cat(inputs, self.cand_x.cuda())
                dist_y = self.cand_y.cuda()
                self.lock_make.release()

            output = self.model(inputs)
            loss = self.criterion(output[:target.size(0)], target)

            # Compute distillation loss
            if self.mem_class_x != {}:
                loss += self.kl(self.lsm(output[y.size(0):]), self.sm(dist_y))

            if training:
                # accumulate gradient
                loss.backward()
                # SGD step
                self.optimizer.step()
                self.training_steps += 1

            outputs.append(output.detach())
            total_loss += float(loss)

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss


    """
    Forward pass for the current epoch
    """
    def forward(self, data_loader, average_output=False, training=False):
        meters = {metric: AverageMeter()
                  for metric in ['loss', 'prec1', 'prec5']}

        # Distillation
        if self.mem_class_x != {}:
            self.cand_x = torch.zeros([self.num_candidates] + list(self.mem_x[0].size())).share_memory_()
            self.cand_y = torch.zeros([self.num_candidates] + list(self.mem_y[0].size())).share_memory_()

            self.lock_make = mp.Lock()
            self.lock_made = mp.Lock()
            self.lock_made.acquire()

            self.p = mp.Process(target=make_candidates, args=(self.num_candidates, self.mem_x, self.mem_y, cand_x, cand_y, lock_make, lock_made, len(self.train_data.get_loader())))
            self.p.start()

        for i_batch, item in enumerate(data_loader):
            inputs = item[0] # x
            target = item[1] # y

            output, loss = self._step(i_batch,
                                      inputs,
                                      target,
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
                                 self.epoch, i_batch, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))

        # Distillation
        if self.mem_class_x != {}:
            self.p.join()

        meters = {name: meter.avg for name, meter in meters.items()}
        meters['error1'] = 100. - meters['prec1']
        meters['error5'] = 100. - meters['prec5']

        return meters


    """
    def forward(self, x):
        self.model.eval()
        ns = x.size(0)

        with torch.no_grad():
            classpred = torch.LongTensor(ns)
            preds = self.model(x)[0].detach().clone().cuda()
            dist = torch.cdist(preds.view(1, *preds.size()), self.mean_features.view(1, *self.mean_features.size())).view(ns, len(self.mem_class_means.keys()))

            for ss in range(ns):
                classpred[ss] = torch.argmin(dist[ss])

            out = torch.zeros(ns, self.num_classes).cuda()
            for ss in range(ns):
                out[ss, classpred[ss]] = 1

            return out
    """


    def update_examplars(self, nc):
        self.eval()
        with torch.no_grad():
            # Reduce exemplar set by updating value of num. exemplars per class
            self.num_exemplars = int(self.num_memories / (len(nc) + len(self.mem_class_x.keys())))

            for c in self.mem_class_x.keys():
                self.mem_class_x[c] = self.mem_class_x[c][:self.num_exemplars]
                self.mem_class_y[c] = self.mem_class_y[c][:self.num_exemplars]

            for c in self.nc:
                # Find indices of examples of classse c
                indxs = (self.mem_y == c).nonzero(as_tuple=False).squeeze()

                # Select examples of classse c
                mem_x_c = torch.index_select(self.mem_x, 0, indxs)

                if not self.candle:
                    # Compute feature vectors of examples of classse c
                    memf_c, _ = self.model(mem_x_c)

                    # Compute the mean feature vector of classse              
                    nb_samples = torch.tensor(memf_c.size(0))
                    sum_memf_c = memf_c.sum(0)

                    sum_memf_c = hvd.allreduce(sum_memf_c, name=f'sum_memf_{c}', op=hvd.Sum)
                    sum_nb_samples = hvd.allreduce(nb_samples, name=f'sum_nb_samples_{c}', op=hvd.Sum)

                    mean_memf_c = sum_memf_c / sum_nb_samples
                    mean_memf_c = mean_memf_c.view(1, self.nb_features)

                    # Compute the distance between each feature vector of the examples of classse c and the mean feature vector of classse c
                    dist_memf_c = torch.cdist(memf_c, mean_memf_c)
                    dist_memf_c = dist_memf_c.view(dist_memf_c.size(0))

                    # Find the indices the self.num_exemplars features vectors closest to the mean feature vector of classse c
                    indices = torch.sort(dist_memf_c)[1][:self.num_exemplars]

                    # Save the self.num_exemplars examples of class c with the closest feature vector to the mean feature vector of classse c
                    self.mem_class_x[c] = torch.index_select(mem_x_c, 0, indices)
                else:
                    means = []
                    fs = {}

                    for i in range(0, len(mem_x_c), 5):
                        x = mem_x_c[i:min(len(mem_x_c), i + 5)]
                        fs[i], _ = self.model(x)
                        means.append(fs[i].sum(0))

                    mean_memf_c = (torch.stack(means).sum(0) / len(mem_x_c)).view(1, self.nb_features)

                    dist_memf_c = None
                    tmp_mem_x = None
                    torch.cuda.empty_cache()

                    for i in range(0, len(mem_x_c), 5):
                        x = mem_x_c[i:min(len(mem_x_c), i + 5)]

                        dist = torch.cdist(fs[i], mean_memf_c)
                        dist = dist.view(dist.size(0))

                        if dist_memf_c is None:
                            indices = torch.sort(dist)[1][:self.num_exemplars]
                            dist_memf_c = torch.index_select(dist, 0, indices)
                            tmp_mem_x = torch.index_select(x, 0, indices)
                        else :
                            x = torch.cat((x, tmp_mem_x))
                            dist = torch.cat((dist, dist_memf_c))
                            indices = torch.sort(dist)[1][:self.num_exemplars]
                            dist_memf_c = torch.index_select(dist, 0, indices)
                            tmp_mem_x = torch.index_select(x, 0, indices)

                    self.mem_class_x[c] = tmp_mem_x
                    del fs

            # recompute outputs for distillation purposes and means for inference purposes
            self.eval()
            for cc in self.mem_class_x.keys():
                if self.candle:
                    outs = []
                    feats = []
                    for i in range(0, len(self.mem_class_x[cc]), 40):
                        f, o = self.model(self.mem_class_x[cc][i:min(i + 40, len(self.mem_class_x[cc]))])
                        outs.append(o)
                        feats.append(f)
                    tmp_features = torch.cat(feats)
                    self.mem_class_y[cc] = torch.cat(outs)
                else:
                    tmp_features, self.mem_class_y[cc] = self.model(self.mem_class_x[cc])

                nb_samples = torch.tensor(tmp_features.size(0))
                sum_memf_c = tmp_features.sum(0)

                sum_memf_c = hvd.allreduce(sum_memf_c, name=f'sum_memf_{c}', op=hvd.Sum)
                sum_nb_samples = hvd.allreduce(nb_samples, name=f'sum_nb_samples_{c}', op=hvd.Sum)

                self.mem_class_means[cc] = sum_memf_c / sum_nb_samples
                self.mean_features = torch.stack(tuple(self.mem_class_means.values()))

        del tmp_features
        torch.cuda.empty_cache()
        self.mem_x = None
        self.mem_y = None


def icarl(agent_config, model_config, model, optimizer, criterion, cuda, log_interval):
    model_name = agent_config['model']
    depth = model_config.get('depth', 18)

    if model_name == 'resnet':
        model_config.setdefault('num_classes', 200)
        model_config.setdefault('num_features', 50)
        agent_config.setdefault('num_classes', 200)
        agent_config.setdefault('num_features', 50)
        agent_config.setdefault('num_representatives', 0)
        agent_config['num_features'] = agent_config['num_features'] * (8 if depth < 50 else 32)
        return icarl_agent(model, agent_config, optimizer, criterion, cuda, log_interval)

    elif model_name == 'mnistnet':
        agent_config.setdefault('num_classes', 200)
        agent_config.setdefault('num_features', 50)
        agent_config.setdefault('num_representatives', 0)
        return icarl_agent(model, agent_config, optimizer, criterion, cuda, log_interval)

    elif model_name == 'candlenet':
        agent_config.setdefault('num_classes', 20)
        agent_config.setdefault('num_features', 50)
        agent_config.setdefault('num_representatives', 0)
        return icarl_agent(model, agent_config, optimizer, criterion, cuda, log_interval)

    else:
        raise ValueError('Unknown model')
