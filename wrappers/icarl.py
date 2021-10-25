# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from continuum.tasks import split_train_val

from utils import *
from models import *


def make_candidates(n, mem_x, mem_y, cand_x, cand_y, lock_make, lock_made, nb_batches):
    for i in range(nb_batches):
        lock_make.acquire()

        selection = torch.randperm(len(mem_x))[n].clone()
        
        nx = mem_x[selection].clone() - cand_x
        ny = mem_y[selection].clone() - cand_y

        cand_x += nx
        cand_y += ny

        lock_made.release()


class icarl_wrapper(nn.Module):
    # Re-implementation of
    # S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert.
    # iCaRL: Incremental classifier and representation learning.
    # CVPR, 2017.
    def __init__(self, model, config, state_dict=None):
        super(icarl_wrapper, self).__init__()

        self.model = model

        # Modified parameters
        self.num_memories = config.get('num_representatives') * config.get('num_classes')
        self.num_exemplars = 0
        self.num_features = config.get('num_features')
        self.num_classes = config.get('num_classes')
        #self.learning_rate = config.get('learning_rate')
        self.num_candidates = config.get('num_candidates')
        #self.candle = self.num_classes == 2

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        # setup distillation losses
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.lsm = nn.LogSoftmax(dim=1)
        self.sm = nn.Softmax(dim=1)

        # memory
        self.memx = None  # stores raw inputs, PxD
        self.memy = None
        self.mem_class_x = {}  # stores exemplars class by class
        self.mem_class_y = {} 
        self.mem_class_means = {}

        self.first_task = True
        self.val_set = None


    def get_state_dict(self):
        return self.model.state_dict()


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


    def update_examplars(self, nc):
        self.eval()
        with torch.no_grad():
            # Reduce exemplar set by updating value of num. exemplars per class
            self.num_exemplars = int(self.num_memories / (len(nc) + len(self.mem_class_x.keys())))

            for c in self.mem_class_x.keys():
                self.mem_class_x[c] = self.mem_class_x[c][:self.num_exemplars]
                self.mem_class_y[c] = self.mem_class_y[c][:self.num_exemplars]

            for c in nc:
                # Find indices of examples of classse c
                indxs = (self.memy == c).nonzero(as_tuple=False).squeeze()

                # Select examples of classse c
                memx_c = torch.index_select(self.memx, 0, indxs)

                if not self.candle:
                    # Compute feature vectors of examples of classse c
                    memf_c, _ = self.model(memx_c)

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
                    self.mem_class_x[c] = torch.index_select(memx_c, 0, indices)
                else:
                    means = []
                    fs = {}

                    for i in range(0, len(memx_c), 5):
                        x = memx_c[i:min(len(memx_c), i + 5)]
                        fs[i], _ = self.model(x)
                        means.append(fs[i].sum(0))

                    mean_memf_c = (torch.stack(means).sum(0) / len(memx_c)).view(1, self.nb_features)

                    dist_memf_c = None
                    tmp_mem_x = None

                    torch.cuda.empty_cache()

                    for i in range(0, len(memx_c), 5):
                        x = memx_c[i:min(len(memx_c), i + 5)]
                        
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

        self.memx = None
        self.memy = None


    def should_distill(self):
        return self.mem_class_x != {}


    def test(self, incremental_test_taskset, task_id):
        self.cuda()

        tot = 0
        test_accuracy = torch.zeros(1)
        test_task = 0

        self.trace[f"Task {task_id}"]['Test'] = {}

        for test_taskset in incremental_test_taskset:
            test_task += 1
            curr_acc = test_accuracy.clone().detach()
            curr_tot = tot

            # Partition the testing data between the workers
            test_sampler, test_loader = distribute(test_taskset)

            self.eval()

            for x, y, _t in test_loader:
                x, y = x.cuda(), y.cuda()
                output = self(x)
                pred = output.data.max(1, keepdim=True)[1]
                pred = torch.flatten(pred).cpu().numpy()
                y = y.cpu().numpy()

                for i in range(len(y)):
                    tot += 1
                    test_accuracy += 1 if pred[i] == y[i] else 0

            task_acc = metric_average((test_accuracy - curr_acc), (tot - curr_tot), f"avg_task{test_task}_acc")

            self.trace[f"Task {task_id}"]['Test'][f"Task {test_task} acc"] = task_acc

        # Average metric values across workers
        accuracy = metric_average(test_accuracy, tot, 'avg_acc')
        self.trace[f"Task {task_id}"]['Test']["acc"] = accuracy

        hprint(f"Accuracy over all tasks seen so far: {accuracy}")


def icarl(config):
    model = config.pop('model', 'resnet')
    depth = config.get('depth', 18)

    config.setdefault('num_classes', 200)
    #config.setdefault('num_representatives', 0)
    #config.setdefault('num_candidates', 10)

    if model == 'resnet':
        config['num_features'] = config.get('num_features', 50) * (8 if depth < 50 else 32)
        return icarl_wrapper(icarl_resnet(config), config)

    elif model == 'mnistnet':
        return icarl_wrapper(icarl_mnistnet(config), config)

    elif model == 'candlenet':
        config.setdefault('num_classes', 20)
        return icarl_wrapper(icarl_candlenet(), config)

    else:
        raise ValueError('Unknown model')
