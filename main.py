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
from utils.log import setup_logging, ResultsLog
from utils.meters import AverageMeter
from utils.model import save_checkpoint
from utils.yaml_params import YParams
from utils.utils import move_cuda

backbone_model_names = sorted(
    name
    for name in backbone.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(backbone.__dict__[name])
)

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(
    description="Distributed deep/continual learning with Horovod + PyTorch"
)
parser.add_argument(
    "--yaml-config",
    default="config.yaml",
    type=str,
    help="path to yaml file containing training configs",
)
parser.add_argument(
    "--config",
    default="",
    type=str,
    help="name of desired config in yaml file",
)
parser.add_argument(
    "--dataset", metavar="DATASET", default="mnist", help="dataset name"
)
parser.add_argument(
    "--dataset-dir",
    default="./datasets",
    help="location of the training dataset in the local filesystem (will be downloaded if needed)",
)
parser.add_argument(
    "--tasksets-config",
    default="",
    help="additional taskset configuration (useful for continual learning)",
)
parser.add_argument(
    "--shard",
    action="store_true",
    default=False,
    help="sample data from a same subset of the dataset at each epoch",
)
parser.add_argument(
    "--backbone",
    metavar="BACKBONE",
    default="resnet50",
    choices=backbone_model_names,
    help="available backbone models: " + " | ".join(backbone_model_names),
)
parser.add_argument(
    "--backbone-config", default="{}", help="additional backbone configuration"
)
parser.add_argument(
    "--model",
    metavar="MODEL",
    default="er",
    choices=model_names,
    help="available models: " + " | ".join(model_names),
)
parser.add_argument(
    "--model-config",
    default="{}",
    help="model configuration",
)
parser.add_argument(
    "--buffer-config",
    default="{}",
    help="rehearsal buffer configuration",
)
parser.add_argument(
    "--load-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="input batch size for training (default: 64)",
)
parser.add_argument(
    "--eval-batch-size",
    type=int,
    default=-1,
    metavar="N",
    help="input batch size for testing (default: same as training)",
)
parser.add_argument(
    "--dataloader-workers",
    type=int,
    default=0,
    help="number of dataloaders workers to spawn",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=25,
    metavar="N",
    help="number of epochs to train (default: 25)",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=5,
    metavar="N",
    help="number of epochs to warmup LR, if scheduler supports (default: 5)",
)
parser.add_argument(
    "--training-only",
    action="store_true",
    help="don't validate after every epoch, only after training on a new task",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="learning rate for a single GPU (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.5,
    metavar="M",
    help="SGD momentum (default: 0.5)",
)
parser.add_argument(
    "--optimizer-regime",
    default="",
    help="optimizer regime, as an array of dicts containing the epoch key",
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="disables CUDA training",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    metavar="S",
    help="random seed (default: 42)",
)
parser.add_argument("--log-level", default="info", help="logging level")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument(
    "--use-dali",
    action="store_true",
    default=False,
    help="use DALI to load data",
)
parser.add_argument(
    "--use-amp",
    action="store_true",
    default=False,
    help="enable Automatic Mixed Precision training",
)
parser.add_argument(
    "--fp16-dali",
    action="store_true",
    default=False,
    help="load images in half precision",
)
parser.add_argument(
    "--fp16-allreduce",
    action="store_true",
    default=False,
    help="use fp16 compression during allreduce",
)
parser.add_argument(
    "--gradient-predivide-factor",
    type=float,
    default=1.0,
    help="apply gradient predivide factor in optimizer (default: 1.0)",
)
parser.add_argument(
    "--weight-decay",
    "--wd",
    type=float,
    default=0,
    metavar="W",
    help="weight decay (default: 0)",
)
parser.add_argument(
    "--results-dir",
    metavar="RESULTS_DIR",
    default="./results",
    help="results dir",
)
parser.add_argument("--save-dir", metavar="SAVE_DIR", default="", help="saved folder")


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.evaluate = not args.training_only

    mp.set_start_method("spawn")

    params = vars(args)
    if args.config:
        yparams = YParams(os.path.abspath(args.yaml_config), args.config)
        for k, v in params.items():
            yparam = yparams[k]
            if yparam:
                params[k] = yparam
                if (
                    k == "buffer_config"
                    or k == "backbone_config"
                    or k == "tasksets_config"
                    or k == "optimizer_regime"
                ):
                    if v:
                        params[k] = str(literal_eval(v) | literal_eval(yparam))
    args = Namespace(**params)

    # Horovod: initialize library.
    hvd.init()
    args.gpus = hvd.size()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    save_path = ""
    if hvd.rank() == 0:
        wandb.init(project="distributed-continual-learning")
        run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{wandb.run.name}"
        if args.model is not None:
            run_name = f"{args.model}-{run_name}"
        wandb.run.name = run_name

        if args.save_dir == "":
            args.save_dir = run_name
        save_path = path.join(args.results_dir, args.save_dir)
        if not path.exists(save_path):
            makedirs(save_path)

        with open(path.join(save_path, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
            wandb.save(path.join(save_path, "args.json"))
        wandb.config.update(args)

        setup_logging(
            path.join(save_path, "log.txt"),
            level=args.log_level,
            dummy=hvd.local_rank() > 0,
        )
        logging.info(f"Saving to {save_path}")

    device = "GPU" if args.cuda else "CPU"
    logging.info(f"Number of {device}s: {hvd.size()}")
    logging.info(f"Run arguments: {args}")

    xp = Experiment(args, save_path)
    xp.run()
    wandb.finish()

    logging.info("Done 🎉🎉🎉")
    sys.exit(0)


class Experiment:
    resume_from_task = 0
    resume_from_epoch = 0

    def __init__(self, args, save_path=""):
        self.args = args
        self.save_path = save_path

        total_num_classes = self.prepare_dataset()

        batch_metrics_path = path.join(self.save_path, "batch_metrics")
        batch_metrics = ResultsLog(
            batch_metrics_path,
            title="Batch metrics - %s" % self.args.save_dir,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0,
        )
        self.create_model(total_num_classes, batch_metrics)

    def create_model(self, total_num_classes, batch_metrics=None):
        # -------------------------------------------------------------------------------------------------#

        # --------------------------#
        # ----- BACKBONE MODEL -----#
        # --------------------------#

        # Creating the model
        backbone_config = {
            "num_classes": total_num_classes,
            "lr": self.args.lr,
            "warmup_epochs": self.args.warmup_epochs,
            "num_epochs": self.args.epochs,
            "num_steps_per_epoch": len(self.train_data_regime.get_loader(0)),
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
            rehearsal_ratio = literal_eval(self.args.buffer_config).get(
                "rehearsal_ratio", 30
            )
            rehearsal_size = math.floor(
                self.train_data_regime.total_num_samples
                * rehearsal_ratio
                / 100
                / total_num_classes
                / hvd.size()
            )
            assert (
                rehearsal_size > 0
            ), "Choose rehearsal_ratio so as to to store at least some samples per class on all processes"
            buffer_config |= {"rehearsal_size": rehearsal_size}

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
            self.args.batch_size,
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
        tasksets_config = {"continual": bool(self.args.tasksets_config)}
        if self.args.tasksets_config != "":
            tasksets_config = dict(
                tasksets_config, **literal_eval(self.args.tasksets_config)
            )

        defaults = {
            "dataset": self.args.dataset,
            "dataset_dir": self.args.dataset_dir,
            "distributed": hvd.size() > 1,
            "use_dali": self.args.use_dali,
            "use_dali_cuda": self.args.use_dali and self.args.cuda,
            "fp16_dali": self.args.fp16_dali,
            "pin_memory": True,
            # https://github.com/horovod/horovod/issues/2053
            "num_workers": self.args.dataloader_workers,
            "shard": self.args.shard,
            "continual": tasksets_config.get("continual"),
            "scenario": tasksets_config.get("scenario", "class"),
            "initial_increment": tasksets_config.get("initial_increment", -1),
            "increment": tasksets_config.get("increment", -1),
            "num_tasks": tasksets_config.get("num_tasks", None),
            "concatenate_tasksets": tasksets_config.get("concatenate_tasksets", False),
        }

        self.train_data_regime = DataRegime(
            hvd,
            {
                **defaults,
                "split": "train",
                "batch_size": self.args.batch_size,
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
            self.model,
            self.train_data_regime,
            self.validate_data_regime,
            self.args.epochs,
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


def on_exit(sig, frame):
    logging.info("Interrupted")
    wandb.finish()
    os.system(
        "kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')"
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    main()
