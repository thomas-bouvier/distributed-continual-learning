import copy
import horovod.torch as hvd
import logging
import numpy as np
import time
import wandb

from tqdm import tqdm

from evaluate.evaluate import evaluate_one_epoch
from utils.log import get_logging_level
from utils.meters import AverageMeter, get_timer
from utils.utils import move_cuda


def measure_performance(step):
    if -1 in step.values():
        return False
    return (
        step["task_id"] == 0
        and step["epoch"] == 0
        and step["batch"] >= 100
        and step["batch"] < 200
    )


def train_one_epoch(model, loader, task_id, epoch, log_interval=10):
    """Forward pass for the current epoch"""
    device = model._device()
    model.backbone.train()

    prefix = "train"
    meters = {
        metric: AverageMeter(f"{prefix}_{metric}")
        for metric in ["loss", "prec1", "prec5", "num_samples", "local_rehearsal_size"]
    }

    batch = 0
    step = dict(task_id=task_id, epoch=epoch, batch=-1)

    epoch_time = 0
    last_batch_time = 0

    enable_tqdm = get_logging_level() in ("info") and hvd.rank() == 0
    with tqdm(
        total=len(loader),
        desc=f"Task #{task_id + 1} {prefix} epoch #{epoch + 1}",
        disable=not enable_tqdm,
    ) as progress:
        timer = get_timer(
            "load",
            step,
            batch_metrics=model.batch_metrics,
            previous_iteration=True,
            dummy=not measure_performance(step),
        )
        timer.__enter__()
        start_batch_time = time.perf_counter()

        for x, y, _ in loader:
            x, y = x.to(device), y.long().to(device)
            timer.__exit__(None, None, None)

            step = dict(task_id=task_id, epoch=epoch, batch=batch)
            model.train_one_step(x, y, meters, step)

            last_batch_time = time.perf_counter() - start_batch_time
            epoch_time += last_batch_time

            if hvd.rank() == 0:
                # Performance metrics
                if (
                    measure_performance(dict(task_id=task_id, epoch=epoch, batch=batch))
                    and model.batch_metrics is not None
                ):
                    model.batch_metrics.save()

            # Logging
            if batch % log_interval == 0 or batch == len(loader):
                logging.debug(
                    "{0}: epoch: {1} [{2}/{3}]\t"
                    "Loss {meters[loss].avg:.4f}\t"
                    "Prec@1 {meters[prec1].avg:.3f}\t"
                    "Prec@5 {meters[prec5].avg:.3f}\t".format(
                        prefix,
                        epoch + 1,
                        batch % len(loader) + 1,
                        len(loader),
                        meters=meters,
                    )
                )

                if measure_performance(dict(task_id=task_id, epoch=epoch, batch=batch)):
                    metrics = model.batch_metrics.get(
                        dict(task_id=task_id, epoch=epoch, batch=batch)
                    )
                    logging.debug(f"batch {batch} time {last_batch_time} sec")
                    logging.debug(
                        f"\t[Python] batch load time {metrics.get('load_time', -1)} sec ({metrics.get('load_time', -1)*100/last_batch_time}%)"
                    )
                    logging.debug(
                        f"\t[Python] batch train time {metrics.get('train_time', -1)} sec ({metrics.get('train_time', -1)*100/last_batch_time}%)"
                    )
                    logging.debug(
                        f"\t[Python] batch accumulate time {metrics.get('accumulate_time', -1)} sec ({metrics.get('accumulate_time', -1)*100/last_batch_time}%)"
                    )

                    if model.use_memory_buffer:
                        logging.debug(
                            f"\t[Python] batch wait time {metrics.get('wait_time', -1)} sec ({metrics.get('wait_time', -1)*100/last_batch_time}%)"
                        )
                        logging.debug(
                            f"\t[Python] batch assemble time {metrics.get('assemble_time', -1)} sec ({metrics.get('assemble_time', -1)*100/last_batch_time}%)"
                        )
                        logging.debug(
                            f"\t[C++] batch copy time {metrics.get('batch_copy_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] bulk prepare time {metrics.get('bulk_prepare_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] rpcs resolve time {metrics.get('rpcs_resolve_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] representatives copy time {metrics.get('representatives_copy_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] buffer update time {metrics.get('buffer_update_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] local_rehearsal_size {meters['local_rehearsal_size'].val.item()}"
                        )

            tqdm_postfix = dict(
                loss=meters["loss"].avg.item(),
                accuracy=meters["prec1"].avg.item(),
            )
            progress.set_postfix(
                {
                    "loss": meters["loss"].avg.item(),
                    "accuracy": meters["prec1"].avg.item(),
                    "num_samples": meters["num_samples"].val.item(),
                    "local_rehearsal_size": meters["local_rehearsal_size"].val.item(),
                }
            )
            progress.update(1)

            batch += 1

            timer = get_timer(
                "load",
                step,
                batch_metrics=model.batch_metrics,
                previous_iteration=True,
                dummy=not measure_performance(step),
            )
            timer.__enter__()
            start_batch_time = time.perf_counter()

    meters["loss"] = meters["loss"].avg.item()
    meters["prec1"] = meters["prec1"].avg.item()
    meters["prec5"] = meters["prec5"].avg.item()
    meters["num_samples"] = meters["num_samples"].val.item()
    meters["local_rehearsal_size"] = meters["local_rehearsal_size"].val.item()
    meters["error1"] = 100.0 - meters["prec1"]
    meters["error5"] = 100.0 - meters["prec5"]
    meters["time"] = epoch_time
    meters["batch"] = batch

    logging.info(f"\nCUMULATED VALUES:")
    logging.info(f"\tlocal_rehearsal_size {meters['local_rehearsal_size']}")
    logging.info(f"epoch time {epoch_time} sec")
    """
    logging.info(
        f"\tepoch load time {self.epoch_load_time} sec ({self.epoch_load_time*100/epoch_time}%)")
    logging.info(
        f"\tepoch train time {self.epoch_train_time} sec ({self.epoch_train_time*100/epoch_time}%)")
    logging.info(
        f"\tepoch wait time {self.epoch_wait_time} sec ({self.epoch_wait_time*100/epoch_time}%)")
    logging.info(
        f"\tepoch assemble time {self.epoch_assemble_time} sec ({self.epoch_assemble_time*100/epoch_time}%)")
    """

    return meters


def train(
    model,
    train_data_regime,
    validate_data_regime,
    epochs,
    resume_from_task=0,
    resume_from_epoch=0,
    evaluate=True,
    log_interval=10,
    dl_metrics=None,
    tasks_metrics=None,
    time_metrics=None,
):
    """
    Train a model on multiple tasks.

    epochs: a number of epochs, or a list of number of epochs if you want to be
    task-specific
    """
    num_tasks = len(train_data_regime.tasksets)
    if not isinstance(epochs, list):
        epochs = [epochs] * num_tasks
    if len(epochs) < num_tasks:
        epochs += (
            [epochs[-1]] * (num_tasks - len(epochs)) if len(epochs) < num_tasks else []
        )

    global_epoch = 0
    global_batch = 0
    img_secs = []
    evaluate_durations = []

    total_start = time.perf_counter()
    model.before_all_tasks(train_data_regime)

    for task_id in range(resume_from_task, num_tasks):
        start = time.perf_counter()
        training_time = time.perf_counter()
        task_metrics = {
            "task_id": task_id,
            "test_tasks_metrics": [],
            "task_averages": [],
        }

        model.before_every_task(task_id, train_data_regime)

        for epoch in range(resume_from_epoch, epochs[task_id]):
            logging.info(
                f"TRAINING on task {task_id + 1}/{num_tasks}, epoch: {epoch + 1}/{epochs[task_id]}, {hvd.size()} device(s)"
            )

            # Horovod: set epoch to sampler for shuffling
            train_data_regime.set_epoch(epoch)
            train_results = train_one_epoch(
                model,
                train_data_regime.get_loader(task_id),
                task_id,
                epoch,
                log_interval=log_interval,
            )

            if hvd.rank() == 0:
                prefix = "train"
                wandb.log(
                    {
                        "epoch": global_epoch,
                        f"{prefix}_loss": train_results["loss"],
                        f"{prefix}_prec1": train_results["prec1"],
                        f"{prefix}_prec5": train_results["prec5"],
                        "lr": model.optimizer_regime.get_lr()[0],
                        "local_rehearsal_size": train_results["local_rehearsal_size"],
                    }
                )

            # evaluate on test set
            before_evaluate_time = time.perf_counter()
            meters = []
            if evaluate or epoch + 1 == epochs[task_id]:
                for test_task_id in range(0, task_id + 1):
                    meters.append(
                        {
                            metric: AverageMeter(f"task_{metric}")
                            for metric in ["loss", "prec1", "prec5"]
                        }
                    )

                for test_task_id in range(0, task_id + 1):
                    logging.info(
                        f"EVALUATING on task {test_task_id + 1}..{task_id + 1}"
                    )

                    validate_data_regime.set_epoch(epoch)
                    validate_results = evaluate_one_epoch(
                        model,
                        validate_data_regime.get_loader(test_task_id),
                        task_id,
                        test_task_id,
                        epoch,
                    )

                    if hvd.rank() == 0 and test_task_id == task_id:
                        prefix = "val"
                        wandb.log(
                            {
                                "epoch": global_epoch,
                                f"{prefix}_loss": validate_results["loss"],
                                f"{prefix}_prec1": validate_results["prec1"],
                                f"{prefix}_prec5": validate_results["prec5"],
                            }
                        )

                    meters[test_task_id]["loss"] = validate_results["loss"]
                    meters[test_task_id]["prec1"] = validate_results["prec1"]
                    meters[test_task_id]["prec5"] = validate_results["prec5"]

                    if hvd.rank() == 0:
                        logging.info(
                            "RESULTS: Testing loss: {validate[loss]:.4f}\n".format(
                                validate=validate_results
                            )
                        )

                    task_metrics_values = dict(test_task_id=test_task_id, epoch=epoch)
                    task_metrics_values.update(
                        {k: v for k, v in validate_results.items()}
                    )
                    task_metrics["test_tasks_metrics"].append(task_metrics_values)

                # meters contains metrics for one epoch (average over all mini-batches)
                averages = {
                    metric: sum(meters[i][metric] for i in range(task_id + 1))
                    / (task_id + 1)
                    for metric in ["loss", "prec1", "prec5"]
                }

                task_metrics_averages = dict(epoch=epoch)
                task_metrics_averages.update({k: v for k, v in averages.items()})
                task_metrics["task_averages"].append(task_metrics_averages)

                if hvd.rank() == 0:
                    wandb.log(
                        {
                            "epoch": global_epoch,
                            "continual_task1_val_loss": meters[0]["loss"],
                            "continual_task1_val_prec1": meters[0]["prec1"],
                            "continual_task1_val_prec5": meters[0]["prec5"],
                        }
                    )
                    wandb.log(
                        {
                            "epoch": global_epoch,
                            "continual_val_loss": averages["loss"],
                            "continual_val_prec1": averages["prec1"],
                            "continual_val_prec5": averages["prec5"],
                        }
                    )

                # TODO: maybe we should compare the averaged loss on all previous tasks?
                if meters[task_id]["loss"] < model.minimal_eval_loss:
                    logging.debug(
                        f"Saving best model with minimal eval loss ({meters[task_id]['loss']}).."
                    )
                    model.minimal_eval_loss = meters[task_id]["loss"]
                    model.best_model = copy.deepcopy(model.backbone.state_dict())

            evaluate_durations.append(time.perf_counter() - before_evaluate_time)

            global_epoch += 1
            global_batch += train_results["batch"]

            if hvd.rank() == 0:
                img_sec = train_results["num_samples"] / train_results["time"]
                img_secs.append(img_sec)
                logging.info(
                    "\nRESULTS: Time taken for epoch {} on {} device(s) is {} sec\n"
                    "Average: {} samples/sec per device\n"
                    "Average on {} device(s): {} samples/sec\n"
                    "Training loss: {train[loss]:.4f}\n".format(
                        epoch + 1,
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
                    epoch=global_epoch,
                    batch=global_batch,
                )
                dl_metrics_values.update(
                    {"train_" + k: v for k, v in train_results.items()}
                )
                dl_metrics_values.update(
                    {
                        "train_img_sec": img_sec,
                        "train_total_img_sec": img_sec * hvd.size(),
                        "train_cumulated_time": time.perf_counter()
                        - total_start
                        - sum(evaluate_durations),
                    }
                )
                if dl_metrics is not None:
                    dl_metrics.add(**dl_metrics_values)
                    dl_metrics.save()

                # Time metrics
                wandb.log(
                    {
                        "epoch": global_epoch,
                        "train_time": dl_metrics_values["train_time"],
                        "train_total_img_sec": dl_metrics_values["train_total_img_sec"],
                        "train_cumulated_time": dl_metrics_values[
                            "train_cumulated_time"
                        ],
                    }
                )

        end = time.perf_counter()

        task_metrics.update({"time": end - start})
        if tasks_metrics is not None:
            tasks_metrics.add(**task_metrics)
            tasks_metrics.save()

        model.after_every_task(task_id, train_data_regime)

    model.after_all_tasks()
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
            "Average: %.1f +-%.1f samples/sec per device" % (img_sec_mean, img_sec_conf)
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
        if time_metrics is not None:
            time_metrics.add(**values)
            time_metrics.save()
