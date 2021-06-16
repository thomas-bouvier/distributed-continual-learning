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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Experiment():
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

        self.timer = Timer()

        self._prepare_dataset()
        self._create_model()


    def _prepare_dataset(self):
            data_dir = self.args.data_dir or './data'
            with FileLock(os.path.expanduser("~/.horovod_lock")):
                train_dataset = \
                    datasets.MNIST(data_dir, train=True, download=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))

            # Horovod: use DistributedSampler to partition the training data.
            self.timer.start('distribute_data')
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.args.batch_size, sampler=self.train_sampler, **self.kwargs)
            self.timer.end('distribute_data')

            test_dataset = \
                datasets.MNIST(data_dir, train=False, download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
            # Horovod: use DistributedSampler to partition the test data.
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.test_batch_size,
                                                    sampler=self.test_sampler, **self.kwargs)


    def _create_model(self):
        self.model = Net()

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

        # Horovod: scale learning rate by lr_scaler.
        self.timer.start('create_optimizer')
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr * lr_scaler,
                            momentum=self.args.momentum)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if self.args.fp16_allreduce else hvd.Compression.none

        # Horovod: wrap optimizer with DistributedOptimizer.
        self.optimizer = hvd.DistributedOptimizer(optimizer,
                                            named_parameters=self.model.named_parameters(),
                                            compression=compression,
                                            op=hvd.Adasum if self.args.use_adasum else hvd.Average,
                                            gradient_predivide_factor=self.args.gradient_predivide_factor)
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
        self.train_sampler.set_epoch(epoch)

        for batch_idx, (data, target) in enumerate(self.train_loader):
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
                    epoch, batch_idx * len(data), len(self.train_sampler),
                    100. * batch_idx / len(self.train_loader), loss.item()))

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


def main(args):
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


if __name__ == '__main__':
    main()