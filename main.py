import argparse
import horovod.torch as hvd
import json
import mlflow
import mlflow.pytorch
import os
import pickle
import time
import torch.multiprocessing as mp
import torch.utils.data.distributed

from ast import literal_eval
from datetime import datetime
from filelock import FileLock
from os import path, makedirs

import models
from cross_entropy import CrossEntropyLoss
from data import DataRegime
from log import ResultsLog
from optimizer import OptimizerRegime
from trainer import Trainer

from torchsummary import summary

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Distributed deep learning with Horovod + PyTorch')
parser.add_argument('--model', metavar='MODEL', required=True,
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--model-config', default='',
                    help='additional architecture configuration')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--eval-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--start-epoch', default=-1, type=int, metavar='N',
                    help='start epoch number, -1 to unset (will start at 0)')
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
parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save-dir', metavar='SAVE_DIR', default='',
                    help='saved folder')

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

    # TODO: check the number of workers
    # https://github.com/horovod/horovod/issues/2053
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    time_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if args.save_dir == '':
        args.save_dir = time_stamp
    save_path = path.join(args.results_dir, args.save_dir)

    if hvd.local_rank() == 0:
        if not path.exists(save_path):
            makedirs(save_path)

    xp = Experiment(save_path, args, kwargs)
    xp.run()


class Experiment():
    def __init__(self, save_path, args, kwargs):
        self.args = args
        self.kwargs = kwargs

        self.save_path = save_path

        self._create_model()
        self._prepare_dataset()

        self.trainer = Trainer(self.model, self.optimizer, self.criterion, args.cuda, args.log_interval)
        self.trainer.training_steps = args.start_epoch * len(self.train_data)


    def _create_model(self):
        model = models.__dict__[self.args.model]

        config = {'dataset': self.args.dataset}
        if self.args.model_config != '':
            config = dict(config, **literal_eval(self.args.model_config))

        self.model = model(**config)

        # By default, Adasum doesn't need scaling up learning rate.
        # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
        lr_scaler = self.args.batches_per_allreduce * hvd.size() if not self.args.use_adasum else 1

        if self.args.cuda:
            #self.timer.start('move_model_to_gpu')
            # Move model to GPU.
            self.model.cuda()
            #self.timer.end('move_model_to_gpu')
            # If using GPU Adasum allreduce, scale learning rate by local_size.
            if self.args.use_adasum and hvd.nccl_built():
                lr_scaler = args.batches_per_allreduce * hvd.local_size()

        #self.timer.start('create_optimizer')
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
        
        # distributed training parameters
        compression = hvd.Compression.fp16 if self.args.fp16_allreduce else hvd.Compression.none
        reduction = hvd.Adasum if self.args.use_adasum else hvd.Average

        self.optimizer = OptimizerRegime(self.model, compression, reduction,
                                    self.args.batches_per_allreduce,
                                    self.args.gradient_predivide_factor,
                                    optim_regime)
        #self.timer.end('create_optimizer')

        # define loss function (criterion) and optimizer
        loss_params = {}
        self.criterion = getattr(model, 'criterion', CrossEntropyLoss)(**loss_params)


    def _prepare_dataset(self):
        self.validate_data = DataRegime(
            getattr(self.model, 'data_eval_regime', None),
            hvd,
            defaults={
                'dataset': self.args.dataset,
                'dataset_dir': self.args.dataset_dir,
                'split': 'validate',
                'batch_size': self.args.eval_batch_size,
                'shuffle': False,
                'distributed': True
            }
        )

        allreduce_batch_size = self.args.batch_size * self.args.batches_per_allreduce
        self.train_data = DataRegime(
            getattr(self.model, 'data_regime', None),
            hvd,
            defaults={
                'dataset': self.args.dataset,
                'dataset_dir': self.args.dataset_dir,
                'split': 'train',
                'batch_size': allreduce_batch_size,
                'shuffle': True,
                'distributed': True
            }
        )


    def run(self):
        with mlflow.start_run():
            # Log our parameters into mlflow
            for key, value in vars(self.args).items():
                mlflow.log_param(key, value)
            mlflow.log_param('gpus', hvd.size())

            results_path = path.join(self.save_path, 'results')
            results = ResultsLog(results_path,
                            title='Training Results - %s' % self.args.save_dir)

            start_epoch = max(self.args.start_epoch, 0)
            self.trainer.training_steps = start_epoch * len(self.train_data)
            for epoch in range(start_epoch, self.args.epochs):
                self.trainer.epoch = epoch

                # Horovod: set epoch to sampler for shuffling.
                self.train_data.set_epoch(epoch)
                self.validate_data.set_epoch(epoch)

                # train for one epoch
                train_results = self.trainer.train(self.train_data.get_loader())

                # evaluate on validation set
                validate_results = self.trainer.validate(self.validate_data.get_loader())

                # Horovod: print output only on first rank.
                if hvd.rank() == 0:
                    print('\nResults: epoch: {0}\n'
                            'Training Loss {train[loss]:.4f} \t\n'
                            'Validation Loss {validate[loss]:.4f} \t\n'
                            .format(epoch + 1, train=train_results,
                            validate=validate_results))

                    values = dict(epoch=epoch + 1, steps=self.trainer.training_steps)
                    values.update({'training ' + k: v for k, v in train_results.items()})
                    values.update({'validation ' + k: v for k, v in validate_results.items()})

                    results.add(**values)
                    results.plot(x='epoch', y=['training loss', 'validation loss'],
                            legend=['training', 'validation'],
                            title='Loss', ylabel='loss')
                    results.plot(x='epoch', y=['training error1', 'validation error1'],
                                legend=['training', 'validation'],
                                title='Error@1', ylabel='error %')
                    results.plot(x='epoch', y=['training error5', 'validation error5'],
                                legend=['training', 'validation'],
                                title='Error@5', ylabel='error %')
                    #results.save()

            # Log the model as an artifact of the MLflow run.
            print("Logging the trained model as a run artifact...")
            mlflow.pytorch.log_model(self.model, artifact_path=f"pytorch-{self.args.model}", pickle_module=pickle)

            return train_results


if __name__ == "__main__":
    main()
