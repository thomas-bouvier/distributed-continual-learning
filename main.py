import argparse
import json
import os
import time

from filelock import FileLock
from utils.timer import Timer
from utils.experiment import Dataset, Model

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

import models
from data import DataRegime
from optimizer import OptimizerRegime

from torchsummary import summary

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Distributed deep learning with PyTorch')
parser.add_argument('--model', metavar='MODEL', required=True,
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='OPT',
                    help='optimizer function')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--weight-decay', '--wd', type=float, default=0,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--dataset', required=True,
                    help="dataset name")
parser.add_argument('--dataset-dir', default='./data',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)


    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    # https://github.com/horovod/horovod/issues/2053
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    xp = Experiment(args, kwargs)
    
    run_duration = time.time()
    run_trace = xp.run()
    run_duration = time.time() - run_duration

    trace = make_head()
    trace = {
        'run_duration' : run_duration,
        'run' : run_trace
    }

    with open(f"trace.json", 'w') as json_file:
        json.dump(trace, json_file, indent=4)


class Experiment():
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

        self.timer = Timer()

        self._create_model()
        self.train_data = self._prepare_dataset()

    def _prepare_dataset(self):
        train_data_defaults = {
            'dataset': self.args.dataset,
            'dataset_dir': self.args.dataset_dir,
            'split': 'train',
            'batch_size': self.args.batch_size,
            'shuffle': True
        }

        return DataRegime(getattr(self.model, 'data_regime', None), hvd, defaults=train_data_defaults)


    def _create_model(self):
        model = models.__dict__[self.args.model]
        config = {'dataset': self.args.dataset}
        self.model = model(**config)

        # By default, Adasum doesn't need scaling up learning rate.
        lr_scaler = hvd.size() if not self.args.use_adasum else 1

        if self.args.cuda:
            self.timer.start('move_model_to_gpu')
            # Move model to GPU.
            self.model.cuda()
            self.timer.end('move_model_to_gpu')
            # If using GPU Adasum allreduce, scale learning rate by local_size.
            if self.args.use_adasum and hvd.nccl_built():
                lr_scaler = hvd.local_size()

        self.timer.start('create_optimizer')
        # Horovod: scale learning rate by lr_scaler.
        optim_regime = getattr(self.model, 'regime', [
            {
                'epoch': 0,
                'optimizer': self.args.optimizer,
                'lr': self.args.lr * lr_scaler,
                'momentum': self.args.momentum,
                'weight_decay': self.args.weight_decay
            }
        ])
        optimizer = OptimizerRegime(self.model, optim_regime,
                self.args.fp16_allreduce,
                self.args.use_adasum,
                self.args.gradient_predivide_factor)
        self.timer.end('create_optimizer')


    def run(self):
        self.timer.start('training')
        for epoch in range(1, self.args.epochs + 1):
            run_trace = self.train(epoch)
        self.timer.end('training')
        return run_trace


    def train(self, epoch: int) -> dict:
        self.model.train()
        self.timer.start(f"epoch_{epoch }")
        # Horovod: set epoch to sampler for shuffling.
        self.train_data.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(self.train_data.get_loader()):
            self.timer.start(f"start_epoch_{epoch}-batch_{batch_idx}")
            self.timer.start(f"epoch_{epoch }-batch_{batch_idx}")

            if self.args.cuda:
                self.timer.start(f"epoch_{epoch }-batch_{batch_idx}-move_batch_to_gpu")
                data, target = data.cuda(), target.cuda()
                self.timer.end(f"epoch_{epoch }-batch_{batch_idx}-move_batch_to_gpu")

            self.timer.start(f"epoch_{epoch}-batch_{batch_idx}-zero_grad")
            self.optimizer.zero_grad()
            self.timer.end(f"epoch_{epoch }-batch_{batch_idx}-zero_grad")

            self.timer.start(f"epoch_{epoch }-batch_{batch_idx}-forward_pass")
            output = self.model(data)
            self.timer.end(f"epoch_{epoch }-batch_{batch_idx}-forward_pass")

            self.timer.start(f"epoch_{epoch }-batch_{batch_idx}-compute_loss")
            loss = F.nll_loss(output, target)
            self.timer.end(f"epoch_{epoch }-batch_{batch_idx}-compute_loss")

            self.timer.start(f"epoch_{epoch }-batch_{batch_idx}-backward_pass")
            loss.backward()
            self.timer.end(f"epoch_{epoch }-batch_{batch_idx}-backward_pass")

            self.timer.start(f"epoch_{epoch }-batch_{batch_idx}-optimizer_step")
            self.optimizer.step()
            self.timer.end(f"epoch_{epoch }-batch_{batch_idx}-optimizer_step")

            self.timer.end(f"epoch_{epoch }-batch_{batch_idx}")
            self.timer.start(f"end_epoch_{epoch}-batch_{batch_idx}")

            if batch_idx % self.args.log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_data),
                    100. * batch_idx / len(self.train_data.get_loader()), loss.item()))

        self.timer.end(f"epoch_{epoch}")

        return self.timer.retrieve()


    def metric_average(self, val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()


    def test(self):
        model.eval()
        test_loss = 0.
        test_accuracy = 0.
        for data, target in test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

        # Horovod: use test_sampler to determine the number of examples in
        # this worker's partition.
        test_loss /= len(test_sampler)
        test_accuracy /= len(test_sampler)

        # Horovod: average metric values across workers.
        test_loss = metric_average(test_loss, 'avg_loss')
        test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

        self.trace['Test']["acc"] = test_accuracy

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
                test_loss, 100. * test_accuracy))


def make_head():
    scen_head = {
        'name' : __file__
    }

    head = {
        'dataset' : Dataset.MNIST,
        'model': Model.NET,
        'GPUs' : hvd.size(),
        'scenario' : scen_head
    }

    return head

if __name__ == "__main__":
    main()