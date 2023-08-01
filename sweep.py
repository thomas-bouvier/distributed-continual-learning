import wandb
from argparse import ArgumentParser
import argparse
import copy
import horovod.torch as hvd
import json
import logging
import math
import os
import signal
import sys
import subprocess
import time
import torch.multiprocessing as mp
import torch.utils.data.distributed
import wandb

from argparse import Namespace
from ast import literal_eval
from datetime import datetime
from os import path, makedirs

import backbone
import models

from data_regime import DataRegime
from optimizer_regime import OptimizerRegime
from train.train import train
from utils.log import setup_logging, ResultsLog, PerformanceResultsLog

from utils.meters import AverageMeter
from utils.model import save_checkpoint
from utils.yaml_params import YParams
from utils.utils import move_cuda


class Experiment:
    resume_from_task = 0
    resume_from_epoch = 0

    def __init__(self, args, save_path, learning_rate, alpha, beta, batchsize, epochs):
        self.args = args
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.batchsize = batchsize
        self.epochs = epochs

        total_num_classes = self.prepare_dataset()

        batch_metrics_path = path.join(self.save_path, "batch_metrics")
        batch_metrics = PerformanceResultsLog(batch_metrics_path)
        self.create_model(total_num_classes, batch_metrics)

    def create_model(self, total_num_classes, batch_metrics=None):
        # -------------------------------------------------------------------------------------------------#

        # --------------------------#
        # ----- BACKBONE MODEL -----#
        # --------------------------#

        # Creating the model
        backbone_config = {
            "num_classes": total_num_classes,
            "lr": self.learning_rate,
            "warmup_epochs": self.args.warmup_epochs,
            "num_epochs": self.epochs,
            "num_steps_per_epoch": len(self.train_data_regime.get_loader(0)),
            "total_num_samples": self.train_data_regime.total_num_samples,
        }
        if self.args.backbone_config != "":
            backbone_config = dict(
                backbone_config, **literal_eval(self.args.backbone_config)
            )
        backbone_model = getattr(backbone, self.args.backbone)(backbone_config)
        logging.info(
            f"Created backbone model {self.args.backbone} with configuration: {json.dumps(backbone_config, indent=2)}"
        )
        num_parameters = sum([l.nelement() for l in backbone_model.parameters()])
        logging.info(f"Number of parameters: {num_parameters}")
        backbone_model = move_cuda(backbone_model, self.args.cuda)

        # Building the optimizer regime
        if self.args.optimizer_regime != "":
            optimizer_regime_dict = literal_eval(self.args.optimizer_regime)
        else:
            optimizer_regime_dict = getattr(backbone_model, "regime")
        logging.info(f"Optimizer regime: {optimizer_regime_dict}")
        optimizer_regime = OptimizerRegime(
            backbone_model,
            hvd.Compression.fp16 if self.args.fp16_allreduce else hvd.Compression.none,
            hvd.Average,
            self.args.gradient_predivide_factor,
            optimizer_regime_dict,
            self.args.use_amp,
        )

        # -------------------------------------------------------------------------------------------------#

        # ------------------#
        # ----- BUFFER -----#
        # ------------------#

        buffer_config = literal_eval(self.args.buffer_config)
        if bool(buffer_config):
            rehearsal_ratio = buffer_config.pop("rehearsal_ratio", 30)
            budget_per_class = math.floor(
                self.train_data_regime.total_num_samples
                * rehearsal_ratio
                / 100
                / total_num_classes
                / hvd.size()
            )
            assert (
                budget_per_class > 0
            ), "Choose rehearsal_ratio so as to to store at least some samples per class on all processes"
            buffer_config |= {"budget_per_class": budget_per_class}

        # -------------------------------------------------------------------------------------------------#

        # -----------------#
        # ----- MODEL -----#
        # -----------------#

        model_config = literal_eval(self.args.model_config)

        # Creating the continual learning model
        model = getattr(models, self.args.model)
        self.model = model(
            backbone_model,
            optimizer_regime,
            self.args.use_amp,
            self.batchsize,
            model_config,
            buffer_config,
            batch_metrics,
        )
        logging.info(
            f"Created model with buffer configuration: {json.dumps(buffer_config, indent=2)}"
        )

        # -------------------------------------------------------------------------------------------------#

        # ----------------------#
        # ----- CHECKPOINT -----#
        # ----------------------#

        # Loading the checkpoint if given
        resume_from_task = 0
        resume_from_epoch = 0
        if hvd.rank() == 0 and self.args.load_checkpoint:
            if not path.isfile(self.args.load_checkpoint):
                parser.error(f"Invalid checkpoint: {self.args.load_checkpoint}")
            checkpoint = torch.load(self.args.load_checkpoint, map_location="cpu")

            # Load checkpoint
            logging.info(f"Loading model {self.args.load_checkpoint}...")
            self.model.backbone.load_state_dict(checkpoint["state_dict"])
            # optimizer_regime.load_state_dict(checkpoint["optimizer_state_dict"])

            # Broadcast resume information
            resume_from_task = checkpoint["task"]
            resume_from_epoch = checkpoint["epoch"]
            logging.info(
                f"Resuming from task {resume_from_task} epoch {resume_from_epoch}"
            )

        self.resume_from_task = hvd.broadcast(
            torch.tensor(resume_from_task), root_rank=0, name="resume_from_task"
        ).item()
        self.resume_from_epoch = hvd.broadcast(
            torch.tensor(resume_from_epoch), root_rank=0, name="resume_from_epoch"
        ).item()

        # Saving an initial checkpoint
        """
        save_checkpoint(
            {
                "task": 0,
                "epoch": 0,
                "model": self.args.model,
                "state_dict": self.model.model.state_dict(),
                "optimizer_state_dict": self.model.optimizer_regime.state_dict(),
            },
            self.save_path,
            is_initial=True,
            dummy=hvd.rank() > 0,
        )
        """

    def prepare_dataset(self):
        defaults = {
            "dataset": self.args.dataset,
            "dataset_dir": self.args.dataset_dir,
            "pin_memory": True,
            "drop_last": True,
            # https://github.com/horovod/horovod/issues/2053
            "num_workers": self.args.dataloader_workers,
            **literal_eval(self.args.tasksets_config),
        }

        self.train_data_regime = DataRegime(
            hvd,
            {
                **defaults,
                "split": "train",
                "batch_size": self.batchsize,
            },
        )
        logging.info(f"Created train data regime: {str(self.train_data_regime.config)}")

        self.validate_data_regime = DataRegime(
            hvd,
            {
                **defaults,
                "split": "validate",
                "batch_size": self.args.eval_batch_size
                if self.args.eval_batch_size > 0
                else self.args.batch_size,
            },
        )
        logging.info(
            f"Created test data regime: {str(self.validate_data_regime.config)}"
        )

        return self.train_data_regime.total_num_classes

    def run(self):
        dl_metrics_path = path.join(self.save_path, "dl_metrics")
        dl_metrics = ResultsLog(
            dl_metrics_path,
            title="DL metrics - %s" % self.args.save_dir,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0,
        )
        tasks_metrics_path = path.join(self.save_path, "tasks_metrics")
        tasks_metrics = ResultsLog(
            tasks_metrics_path,
            title="Tasks metrics - %s" % self.args.save_dir,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0,
        )
        time_metrics_path = path.join(self.save_path, "time_metrics")
        time_metrics = ResultsLog(
            time_metrics_path,
            title="Time metrics - %s" % self.args.save_dir,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0,
        )

        train(
            self.alpha,
            self.beta,
            self.model,
            self.train_data_regime,
            self.validate_data_regime,
            literal_eval(self.args.tasksets_config).get("epochs", self.args.epochs),
            resume_from_task=self.resume_from_task,
            resume_from_epoch=self.resume_from_epoch,
            log_interval=self.args.log_interval,
            dl_metrics=dl_metrics,
            tasks_metrics=tasks_metrics,
            time_metrics=time_metrics,
        )

        save_checkpoint(
            {
                "task": len(self.train_data_regime.tasksets) - 1,
                "epoch": self.args.epochs - 1,
                "model": self.args.backbone,
                "state_dict": self.model.backbone.state_dict(),
                "optimizer_state_dict": self.model.optimizer_regime.state_dict(),
            },
            self.save_path,
            is_final=True,
            dummy=hvd.rank() > 0,
        )


class Sweep:
    def __init__(self, args, save_path=""):
        self.args = args
        self.save_path = save_path

    def set_config(self):
        sweep_config = {
            "method": "random",
            "metric": {"name": "continual_val_prec1", "goal": "maximize"},
            "parameters": {
                "epochs": {"min": 5, "max": 40},
                "lr": {"min": 0.0001, "max": 0.5},
                "alpha": {"min": 0.1, "max": 0.9},
                "beta": {"min": 0.1, "max": 0.9},
                "batch-size": {"values": [16, 32, 64, 128]},
            },
        }

        sweep_id = wandb.sweep(sweep_config, project="distributed-continual-learning")
        return sweep_id

    def train(self):
        trainer = Experiment(
            self.args,
            self.save_path,
            learning_rate=self.args.lr,
            alpha=self.args.alpha,
            beta=self.args.beta,
            batchsize=self.args.batch_size,
            epochs=self.args.epochs,
        )
        trainer.run()

    def run_sweep_agent(self):
        wandb.agent(self.set_config(), function=self.train())
