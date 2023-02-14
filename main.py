import argparse
import copy
import horovod.torch as hvd
import json
import logging
import numpy as np
import os
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
    "--yaml_config",
    default="config.yaml",
    type=str,
    help="path to yaml file containing training configs",
)
parser.add_argument(
    "--config",
    default="base",
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
    "--buffer-cuda",
    action="store_true",
    default=False,
    help="store replay buffers in CUDA memory rather than cpu",
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
    "--checkpoint",
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
    "--batches-per-allreduce",
    type=int,
    default=1,
    help="number of batches processed locally before "
    "executing allreduce across workers; it multiplies "
    "total batch size.",
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
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)
parser.add_argument(
    "--training-only",
    action="store_true",
    help="don't validate between training phases",
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
    "--optimizer",
    type=str,
    default="SGD",
    metavar="OPT",
    help="optimizer function",
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
    "--use-adasum",
    action="store_true",
    default=False,
    help="use adasum algorithm to do reduction",
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
parser.add_argument("--save-dir", metavar="SAVE_DIR",
                    default="", help="saved folder")
parser.add_argument(
    "--tensorboard",
    action="store_true",
    default=False,
    help="set tensorboard logging",
)
parser.add_argument(
    "--tensorwatch",
    action="store_true",
    default=False,
    help="set tensorwatch logging",
)
parser.add_argument(
    "--tensorwatch-port", default=0, type=int, help="set tensorwatch port"
)


def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.buffer_cuda = args.buffer_cuda and args.cuda
    args.buffer_tensorboard = args.log_buffer and args.tensorboard
    args.evaluate = not args.training_only

    mp.set_start_method("spawn")

    params = vars(args)
    yparams = YParams(os.path.abspath(args.yaml_config), args.config)
    for k, v in params.items():
        yparam = yparams[k]
        if yparam:
            if k == 'model_config' or k == 'agent_config' or k == 'tasksets_config':
                if v :
                    params[k] = str(literal_eval(v) | literal_eval(yparam))
                else:
                    params[k] = yparam
            else:
                params[k] = yparam
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
        if args.agent is not None:
            wandb.init(project=f"distributed-continual-learning_{args.agent}")
        else:
            wandb.init(project="distributed-continual-learning")
        run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{wandb.run.name}"
        wandb.run.name = run_name

        if args.save_dir == "":
            args.save_dir = run_name
        save_path = path.join(args.results_dir, args.save_dir)
        if not path.exists(save_path):
            makedirs(save_path)

        with open(path.join(save_path, "args.json"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
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


class Experiment:
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
        # By default, Adasum doesn't need scaling up learning rate.
        # For sum/average with gradient Accumulation: scale learning rate by batches_per_allreduce
        lr_scaler = (
            self.args.batches_per_allreduce * hvd.size()
            if not self.args.use_adasum
            else 1
        )
        if self.args.cuda:
            # If using GPU Adasum allreduce, scale learning rate by local_size.
            if self.args.use_adasum and hvd.nccl_built():
                lr_scaler = self.args.batches_per_allreduce * hvd.local_size()

        model_name = models.__dict__[self.args.model]
        model_config = {
            "num_classes": total_num_classes,
        }
        if self.args.model_config != "":
            model_config = dict(
                model_config, **literal_eval(self.args.model_config))
        model = model_name(model_config)

        # Building the model
        if self.args.checkpoint:
            if not path.isfile(self.args.checkpoint):
                parser.error(f"Invalid checkpoint: {self.args.checkpoint}")
            checkpoint = torch.load(self.args.checkpoint, map_location="cpu")
            # Overrride configuration with checkpoint info
            model_name = checkpoint.get("model", model_name)
            model_config = checkpoint.get("config", model_config)
            # Load checkpoint
            logging.info(f"Loading model {self.args.checkpoint}..")
            model.load_state_dict(checkpoint["state_dict"])
        save_checkpoint(
            {
                "task": 0,
                "epoch": 0,
                "model": self.args.model,
                "model_config": self.args.model_config,
                "state_dict": model.state_dict(),
            },
            self.save_path,
            is_initial=True,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0,
        )
        logging.info(
            f"Created model {self.args.model} with configuration: {model_config}"
        )
        num_parameters = sum([l.nelement() for l in model.parameters()])
        logging.info(f"Number of parameters: {num_parameters}")
        model = move_cuda(model, self.args.cuda)

        # Building the optimizer regime
        regime = getattr(model, "regime")
        logging.info(f"Optimizer regime: {regime}")
        self.optimizer_regime = OptimizerRegime(
            model,
            self.args.lr * lr_scaler,
            hvd.Compression.fp16 if self.args.fp16_allreduce else hvd.Compression.none,
            hvd.Adasum if self.args.use_adasum else hvd.Average,
            self.args.batches_per_allreduce,
            self.args.gradient_predivide_factor,
            regime,
            self.args.use_amp
        )

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
            self.args.use_amp,
            agent_config,
            self.optimizer_regime,
            self.args.batch_size * self.args.batches_per_allreduce,
            self.args.cuda,
            self.args.log_level,
            self.args.log_buffer,
            self.args.log_interval,
            batch_metrics
        )
        self.agent.epochs = self.args.epochs
        logging.info(f"Created agent with configuration: {agent_config}")

        if self.args.tensorboard:
            self.agent.set_tensorboard_writer(
                save_path=self.save_path,
                images=self.args.buffer_tensorboard,
                dummy=hvd.rank() > 0 or hvd.local_rank() > 0,
            )
        if self.args.tensorwatch:
            self.agent.set_tensorwatch_watcher(
                filename=path.abspath(
                    path.join(self.save_path, "tensorwatch.log")),
                port=self.args.tensorwatch_port,
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
            "initial_increment": tasksets_config.get("initial_increment", 0),
            "increment": tasksets_config.get("increment", 1),
            "num_tasks": tasksets_config.get("num_tasks", None),
            "concatenate_tasksets": tasksets_config.get("concatenate_tasksets", False),
        }

        self.train_data_regime = DataRegime(
            hvd,
            {
                **defaults,
                "split": "train",
                "batch_size": self.args.batch_size * self.args.batches_per_allreduce,
            },
        )
        logging.info("Created train data regime: %s",
                     repr(self.train_data_regime))

        self.validate_data_regime = DataRegime(
            hvd,
            {
                **defaults,
                "split": "validate",
                "batch_size": self.args.eval_batch_size
                if self.args.eval_batch_size > 0
                else self.args.batch_size * self.args.batches_per_allreduce,
            },
        )
        logging.info("Created test data regime: %s",
                     repr(self.validate_data_regime))

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

        total_start = time.time()
        self.agent.before_all_tasks(self.train_data_regime)

        for task_id in range(0, len(self.train_data_regime.tasksets)):
            start = time.time()
            training_time = time.time()
            logging.info(
                "\n==============================\nStarting task %s",
                task_id,
            )

            task_metrics = {"task_id": task_id, "test_task_metrics": []}

            self.train_data_regime.set_task_id(task_id)
            self.agent.before_every_task(task_id, self.train_data_regime)
            self.optimizer_regime.num_steps = len(
                self.train_data_regime.get_loader())

            for i_epoch in range(0, self.args.epochs):
                logging.info(f"Starting task {task_id}, epoch: {i_epoch}")
                self.agent.epoch = i_epoch

                # Horovod: set epoch to sampler for shuffling
                self.train_data_regime.set_epoch(i_epoch)

                # train for one epoch
                train_results = self.agent.train(self.train_data_regime)

                # evaluate on test set
                before_evaluate_time = time.time()
                if self.args.evaluate:
                    meters = {
                        metric: AverageMeter(f"task_{metric}")
                        for metric in ["loss", "prec1", "prec5"]
                    }
                    if self.args.agent == "icarl":
                        self.agent.update_exemplars(
                            self.agent.nc, training=False)
                    for test_task_id in range(0, task_id+1):
                        self.validate_data_regime.set_task_id(test_task_id)
                        self.validate_data_regime.get_loader(True)
                        self.validate_data_regime.set_epoch(i_epoch)

                        validate_results = self.agent.validate(
                            self.validate_data_regime)
                        meters["loss"].update(validate_results["loss"])
                        meters["prec1"].update(validate_results["prec1"])
                        meters["prec5"].update(validate_results["prec5"])

                        if hvd.rank() == 0:
                            logging.info(
                                "\nRESULTS: Testing loss: {validate[loss]:.4f}\n".format(
                                    validate=validate_results
                                )
                            )

                            task_metrics_values = dict(
                                test_task_id=test_task_id + 1, epoch=i_epoch
                            )
                            task_metrics_values.update(
                                {"test_" + k: v for k, v in validate_results.items()}
                            )
                            task_metrics["test_task_metrics"].append(
                                task_metrics_values
                            )

                    if meters["loss"].avg < self.agent.minimal_eval_loss:
                        logging.debug(
                            f"Saving best model with minimal eval loss ({meters['loss'].avg}).."
                        )
                        self.agent.minimal_eval_loss = meters["loss"].avg
                        self.agent.best_model = copy.deepcopy(
                            self.agent.model.state_dict()
                        )

                evaluate_durations.append(time.time() - before_evaluate_time)

                self.agent.after_every_epoch()

                if hvd.rank() == 0:
                    img_sec = (
                        train_results["steps"]
                        * self.args.batch_size
                        / train_results["time"]
                    )
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

                    if hvd.rank() == 0:
                        wandb.log({"epoch": self.agent.global_epoch,
                                   "epoch_time": train_results["time"],
                                   "img_sec": img_sec * hvd.size()})
                    if self.agent.writer is not None:
                        self.agent.writer.add_scalar(
                            "img_sec", img_sec * hvd.size(), self.agent.global_epoch
                        )
                    dl_metrics_values = dict(
                        task_id=task_id,
                        epoch=self.agent.global_epoch,
                        steps=self.agent.global_steps,
                    )
                    dl_metrics_values.update(
                        {"train_" + k: v for k, v in train_results.items()}
                    )
                    dl_metrics_values.update({"train_img_sec": img_sec})
                    dl_metrics_values.update(
                        {"train_total_img_sec": img_sec * hvd.size()}
                    )
                    dl_metrics.add(**dl_metrics_values)
                    """
                    dl_metrics.plot(x='epoch', y=['training loss', 'validation loss'],
                                    legend=['training', 'validation'],
                                    title='Loss', ylabel='loss')
                    dl_metrics.plot(x='epoch', y=['training prec1', 'validation prec1'],
                                    legend=['training', 'validation'],
                                    title='Prec@1', ylabel='prec %')
                    dl_metrics.plot(x='epoch', y=['training prec5', 'validation prec5'],
                                    legend=['training', 'validation'],
                                    title='Prec@5', ylabel='prec %')
                    dl_metrics.plot(x='epoch', y=['training error1', 'validation error1'],
                                    legend=['training', 'validation'],
                                    title='Error@1', ylabel='error %')
                    dl_metrics.plot(x='epoch', y=['training error5', 'validation error5'],
                                    legend=['training', 'validation'],
                                    title='Error@5', ylabel='error %')
                    """
                    dl_metrics.save()

            end = time.time()
            task_metrics.update({"time": end - start})
            # logging.debug(f"\nTask metrics : {task_metrics}")
            tasks_metrics.add(**task_metrics)
            tasks_metrics.save()

            self.agent.after_every_task()

        self.agent.after_all_tasks()
        total_end = time.time()

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
                "total_training_time": total_training_time,
                "training img_sec": img_sec_mean,
                "training total_img_sec": img_sec_mean * hvd.size(),
            }
            time_metrics.add(**values)
            time_metrics.save()

        save_checkpoint(
            {
                "task": len(self.train_data_regime.tasksets),
                "epoch": self.args.epochs,
                "model": self.args.model,
                "model_config": self.args.model_config,
                "state_dict": self.agent.model.state_dict(),
            },
            self.save_path,
            is_final=True,
            dummy=hvd.rank() > 0 or hvd.local_rank() > 0,
        )


if __name__ == "__main__":
    main()
