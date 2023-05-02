import argparse
import copy
import horovod.torch as hvd
import json
import logging
import numpy as np
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

import models
import agents

from data_regime import DataRegime
from optimizer_regime import OptimizerRegime
from utils.log import save_checkpoint, setup_logging, ResultsLog
from utils.meters import AverageMeter
from utils.yaml_params import YParams
from utils.utils import move_cuda

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

agent_names = sorted(
    name
    for name in agents.__dict__
    if name.islower() and not name.startswith("__") and callable(agents.__dict__[name])
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
    "--log-buffer",
    action="store_true",
    default=False,
    help="log replay buffers to tensorboard",
)
parser.add_argument(
    "--agent",
    metavar="AGENT",
    default=None,
    choices=agent_names,
    help="model agent: " + " | ".join(agent_names),
)
parser.add_argument(
    "--agent-config",
    default="",
    help="additional agent configuration"
)
parser.add_argument(
    "--model",
    metavar="MODEL",
    default="mnistnet",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names),
)
parser.add_argument(
    "--model-config",
    default="",
    help="additional model architecture configuration",
)
parser.add_argument(
    "--use-mask",
    action="store_true",
    help="use a continual mask on classes seen so far"
)
parser.add_argument(
    "--load-checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--save-all-checkpoints",
    action="store_true",
    default=False,
    help="save a ceckpoint after every epoch",
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
parser.add_argument(
    "--log-level",
    default='info',
    help="logging level"
)
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
parser.add_argument("--save-dir",
    metavar="SAVE_DIR",
    default="",
    help="saved folder"
)
parser.add_argument(
    "--tensorboard",
    action="store_true",
    default=False,
    help="set tensorboard logging",
)


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.buffer_tensorboard = args.log_buffer and args.tensorboard
    args.evaluate = not args.training_only

    mp.set_start_method("spawn")

    params = vars(args)
    if args.config:
        yparams = YParams(os.path.abspath(args.yaml_config), args.config)
        for k, v in params.items():
            yparam = yparams[k]
            if yparam:
                params[k] = yparam
                if k == 'model_config' or k == 'agent_config' or k == 'tasksets_config' or k == 'optimizer_regime':
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
        if args.agent is not None:
            run_name = f"{args.agent}-{run_name}"
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

        setup_logging(path.join(save_path, "log.txt"),
                      level=args.log_level,
                      dummy=hvd.local_rank() > 0)
        logging.info(f"Saving to {save_path}")

    device = "GPU" if args.cuda else "CPU"
    logging.info(f"Number of {device}s: {hvd.size()}")
    logging.info(f"Run arguments: {args}")

    xp = Experiment(args, save_path)
    xp.run()
    wandb.finish()

    logging.info("Done!")
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
            batch_metrics_path, title="Batch metrics - %s" % self.args.save_dir,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0
        )
        self.create_agent(total_num_classes, batch_metrics)

    def create_agent(self, total_num_classes, batch_metrics=None):
        # Creating the model
        model_name = models.__dict__[self.args.model]
        model_config = {
            "num_classes": total_num_classes,
            "lr": self.args.lr,
            "warmup_epochs": self.args.warmup_epochs,
            "num_epochs": self.args.epochs,
        }
        if self.args.model_config != "":
            model_config = dict(
                model_config, **literal_eval(self.args.model_config))
        model = model_name(model_config, len(self.train_data_regime.get_loader()))
        logging.info(
            f"Created model {self.args.model} with configuration: {json.dumps(model_config, indent=2)}"
        )
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info(f"Number of parameters: {num_parameters}")
        model = move_cuda(model, self.args.cuda)

        # Building the optimizer regime
        if self.args.optimizer_regime != "":
            optimizer_regime_dict = literal_eval(self.args.optimizer_regime)
        else:
            optimizer_regime_dict = getattr(model, "regime")
        logging.info(f"Optimizer regime: {optimizer_regime_dict}")
        optimizer_regime = OptimizerRegime(
            model,
            hvd.Compression.fp16 if self.args.fp16_allreduce else hvd.Compression.none,
            hvd.Average,
            self.args.gradient_predivide_factor,
            optimizer_regime_dict,
            self.args.use_amp
        )

        # Loading the checkpoint if given
        resume_from_task = 0
        resume_from_epoch = 0
        if hvd.rank() == 0 and self.args.load_checkpoint:
            if not path.isfile(self.args.load_checkpoint):
                parser.error(f"Invalid checkpoint: {self.args.load_checkpoint}")
            checkpoint = torch.load(self.args.load_checkpoint, map_location="cpu")

            # Override configuration with checkpoint info
            model_name = checkpoint.get("model", model_name)
            model_config = checkpoint.get("config", model_config)

            # Load checkpoint
            logging.info(f"Loading model {self.args.load_checkpoint}..")
            model.load_state_dict(checkpoint["state_dict"])
            #optimizer_regime.load_state_dict(checkpoint["optimizer_state_dict"])

            # Broadcast resume information
            resume_from_task = checkpoint["task"]
            resume_from_epoch = checkpoint["epoch"]
            logging.info(f"Resuming from task {resume_from_task} epoch {resume_from_epoch}")

        self.resume_from_task = hvd.broadcast(torch.tensor(resume_from_task), root_rank=0,
                    name='resume_from_task').item()
        self.resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                    name='resume_from_epoch').item()

        # Creating the agent
        agent = (
            agents.__dict__[self.args.agent]
            if self.args.agent is not None
            else agents.base
        )
        agent_config = {}
        if self.args.agent_config != "":
            agent_config = dict(
                agent_config, **literal_eval(self.args.agent_config))
        self.agent = agent(
            model,
            self.args.use_mask,
            self.args.use_amp,
            agent_config,
            optimizer_regime,
            self.args.batch_size,
            self.args.cuda,
            self.args.log_level,
            self.args.log_buffer,
            self.args.log_interval,
            batch_metrics
        )
        self.agent.epochs = self.args.epochs
        logging.info(f"Created agent with configuration: {json.dumps(agent_config, indent=2)}")

        # Saving an initial checkpoint
        """
        save_checkpoint(
            {
                "task": 0,
                "epoch": 0,
                "model": self.args.model,
                "model_config": self.args.model_config,
                "state_dict": self.agent.model.state_dict(),
                "optimizer_state_dict": self.agent.optimizer_regime.state_dict(),
            },
            self.save_path,
            is_initial=True,
            dummy=hvd.rank() > 0,
        )
        """

        if self.args.tensorboard:
            self.agent.set_tensorboard_writer(
                save_path=self.save_path,
                images=self.args.buffer_tensorboard,
                dummy=hvd.rank() > 0 or hvd.local_rank() > 0,
            )

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
        logging.info(f"Created test data regime: {str(self.validate_data_regime.config)}")

        return self.train_data_regime.total_num_classes

    def run(self):
        dl_metrics_path = path.join(self.save_path, "dl_metrics")
        dl_metrics = ResultsLog(
            dl_metrics_path, title="DL metrics - %s" % self.args.save_dir,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0
        )
        tasks_metrics_path = path.join(self.save_path, "tasks_metrics")
        tasks_metrics = ResultsLog(
            tasks_metrics_path, title="Tasks metrics - %s" % self.args.save_dir,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0
        )
        time_metrics_path = path.join(self.save_path, "time_metrics")
        time_metrics = ResultsLog(
            time_metrics_path, title="Time metrics - %s" % self.args.save_dir,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0
        )

        img_secs = []
        evaluate_durations = []

        total_start = time.perf_counter()
        self.agent.before_all_tasks(self.train_data_regime)

        for task_id in range(self.resume_from_task, len(self.train_data_regime.tasksets)):
            start = time.perf_counter()
            training_time = time.perf_counter()

            task_metrics = {"task_id": task_id, "test_tasks_metrics": [], "task_averages": []}

            self.train_data_regime.set_task_id(task_id)
            self.agent.before_every_task(task_id, self.train_data_regime)

            for i_epoch in range(self.resume_from_epoch, self.args.epochs):
                logging.info(f"TRAINING on task {task_id + 1}/{len(self.train_data_regime.tasksets)}, epoch: {i_epoch + 1}/{self.args.epochs}, {hvd.size()} device(s)")
                self.agent.epoch = i_epoch

                # Horovod: set epoch to sampler for shuffling
                self.train_data_regime.set_epoch(i_epoch)

                # train for one epoch
                train_results = self.agent.train(self.train_data_regime)

                # evaluate on test set
                before_evaluate_time = time.perf_counter()
                meters = []
                if self.args.evaluate or i_epoch + 1 == self.args.epochs:
                    for test_task_id in range(0, task_id + 1):
                        meters.append({
                            metric: AverageMeter(f"task_{metric}")
                            for metric in ["loss", "prec1", "prec5"]
                        })

                    for test_task_id in range(0, task_id + 1):
                        self.validate_data_regime.set_task_id(test_task_id)
                        self.validate_data_regime.get_loader(True)
                        self.validate_data_regime.set_epoch(i_epoch)

                        logging.info(f"EVALUATING on task {test_task_id + 1}..{task_id + 1}")

                        validate_results = self.agent.validate(self.validate_data_regime, previous_task=(test_task_id != task_id))
                        meters[test_task_id]["loss"] = validate_results["loss"]
                        meters[test_task_id]["prec1"] = validate_results["prec1"]
                        meters[test_task_id]["prec5"] = validate_results["prec5"]

                        if hvd.rank() == 0:
                            logging.info(
                                "RESULTS: Testing loss: {validate[loss]:.4f}\n".format(
                                    validate=validate_results
                                )
                            )

                        task_metrics_values = dict(
                            test_task_id=test_task_id, epoch=i_epoch
                        )
                        task_metrics_values.update(
                            {k: v for k, v in validate_results.items()}
                        )
                        task_metrics["test_tasks_metrics"].append(
                            task_metrics_values
                        )

                    # meters contains metrics for one epoch (average over all mini-batches)
                    averages = {metric: sum(meters[i][metric] for i in range(task_id + 1)) / (task_id + 1)
                            for metric in ["loss", "prec1", "prec5"]}

                    task_metrics_averages = dict(
                        epoch=i_epoch
                    )
                    task_metrics_averages.update(
                        {k: v for k, v in averages.items()}
                    )
                    task_metrics["task_averages"].append(
                        task_metrics_averages
                    )

                    if hvd.rank() == 0:
                        wandb.log({"epoch": self.agent.global_epoch,
                                "continual_task1_val_loss": meters[0]["loss"],
                                "continual_task1_val_prec1": meters[0]["prec1"],
                                "continual_task1_val_prec5": meters[0]["prec5"]})
                        wandb.log({"epoch": self.agent.global_epoch,
                                "continual_val_loss": averages["loss"],
                                "continual_val_prec1": averages["prec1"],
                                "continual_val_prec5": averages["prec5"]})

                    #TODO: maybe we should compared the averaged loss on all previous tasks?
                    if meters[task_id]["loss"] < self.agent.minimal_eval_loss:
                        logging.debug(
                            f"Saving best model with minimal eval loss ({meters[task_id]['loss']}).."
                        )
                        self.agent.minimal_eval_loss = meters[task_id]["loss"]
                        self.agent.best_model = copy.deepcopy(
                            self.agent.model.state_dict()
                        )

                evaluate_durations.append(time.perf_counter() - before_evaluate_time)

                self.agent.after_every_epoch()

                if hvd.rank() == 0:
                    img_sec = train_results["num_samples"] / train_results["time"]
                    img_secs.append(img_sec)
                    logging.info(
                        "\nRESULTS: Time taken for epoch {} on {} device(s) is {} sec\n"
                        "Average: {} samples/sec per device\n"
                        "Average on {} device(s): {} samples/sec\n"
                        "Training loss: {train[loss]:.4f}\n".format(
                            i_epoch + 1,
                            hvd.size(),
                            train_results["time"],
                            img_sec,
                            hvd.size(),
                            img_sec * hvd.size(),
                            train=train_results,
                        )
                    )

                    # DL metrics
                    dl_metrics_values = dict(
                        task_id=task_id,
                        epoch=self.agent.global_epoch,
                        batch=self.agent.global_batch,
                    )
                    dl_metrics_values.update({
                        "train_" + k: v for k, v in train_results.items()
                    })
                    dl_metrics_values.update({
                        "train_img_sec": img_sec,
                        "train_total_img_sec": img_sec * hvd.size(),
                        "train_cumulated_time": time.perf_counter() - total_start - sum(evaluate_durations),
                    })
                    dl_metrics.add(**dl_metrics_values)
                    dl_metrics.save()

                    # Time metrics
                    wandb.log({"epoch": self.agent.global_epoch,
                               "train_time": dl_metrics_values["train_time"],
                               "train_total_img_sec": dl_metrics_values["train_total_img_sec"],
                               "train_cumulated_time": dl_metrics_values["train_cumulated_time"],
                    })
                    if self.agent.writer is not None:
                        self.agent.writer.add_scalar(
                            "img_sec", img_sec * hvd.size(), self.agent.global_epoch
                        )

                if self.args.save_all_checkpoints:
                    save_checkpoint(
                        {
                            "task": task_id,
                            "epoch": i_epoch,
                            "model": self.args.model,
                            "model_config": self.args.model_config,
                            "state_dict": self.agent.model.state_dict(),
                            "optimizer_state_dict": self.agent.optimizer_regime.state_dict()
                        },
                        self.save_path,
                        filename=f"checkpoint_task_{task_id}_epoch_{i_epoch}.pth.tar",
                        dummy=hvd.rank() > 0
                    )

            end = time.perf_counter()
            task_metrics.update({"time": end - start})
            # logging.debug(f"\nTask metrics : {task_metrics}")
            tasks_metrics.add(**task_metrics)
            tasks_metrics.save()

            self.agent.after_every_task()

        self.agent.after_all_tasks()
        total_end = time.perf_counter()

        if hvd.rank() == 0:
            img_sec_mean = np.mean(img_secs)
            img_sec_conf = 1.96 * np.std(img_secs)
            total_time = total_end - total_start
            total_training_time = total_time - sum(evaluate_durations)

            logging.info("\nFINAL RESULTS:")
            logging.info(f"Total time: {total_time}")
            logging.info(f"Total training time: {total_training_time}")
            logging.info(
                "Average: %.1f +-%.1f samples/sec per device"
                % (img_sec_mean, img_sec_conf)
            )
            logging.info(
                "Average on %d device(s): %.1f +-%.1f"
                % (
                    hvd.size(),
                    hvd.size() * img_sec_mean,
                    hvd.size() * img_sec_conf,
                )
            )
            values = {
                "total_time": total_time,
                "total_train_time": total_training_time,
                "train_img_sec": img_sec_mean,
                "train_total_img_sec": img_sec_mean * hvd.size(),
            }
            time_metrics.add(**values)
            time_metrics.save()

        save_checkpoint(
            {
                "task": len(self.train_data_regime.tasksets) - 1,
                "epoch": self.args.epochs - 1,
                "model": self.args.model,
                "model_config": self.args.model_config,
                "state_dict": self.agent.model.state_dict(),
                "optimizer_state_dict": self.agent.optimizer_regime.state_dict()
            },
            self.save_path,
            is_final=True,
            dummy=hvd.rank() > 0
        )


def on_exit(sig, frame):
    logging.info("Interrupted")
    wandb.finish()
    os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    main()
