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


def train_one_epoch(
    model, loader, task_id, epoch, global_batch, global_epoch, scenario, log_interval=10
):
    """Forward pass for the current epoch"""
    model.backbone.train()

    prefix = "train"
    metrics = (
        ["loss", "prec1", "prec5", "num_samples", "local_rehearsal_size"]
        if scenario != "reconstruction"
        else ["loss", "loss_amp", "loss_ph", "num_samples", "local_rehearsal_size"]
    )
    meters = {metric: AverageMeter(f"{prefix}_{metric}") for metric in metrics}

    batch = 0
    step = dict(task_id=task_id, epoch=epoch, batch=-1)

    epoch_time = 0
    last_batch_time = 0

    enable_tqdm = get_logging_level() >= logging.INFO and hvd.rank() == 0
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

        for data in loader:
            timer.__exit__(None, None, None)

            step = dict(
                task_id=task_id, epoch=epoch, batch=batch, global_epoch=global_epoch
            )
            train_fn = (
                model.train_one_step
                if scenario != "reconstruction"
                else model.train_recon_one_step
            )
            train_fn(data, meters, step)

            last_batch_time = time.perf_counter() - start_batch_time
            epoch_time += last_batch_time

            if hvd.rank() == 0:
                # Performance metrics
                if (
                    measure_performance(dict(task_id=task_id, epoch=epoch, batch=batch))
                    and model.batch_metrics is not None
                ):
                    model.batch_metrics.save()

                wandb.log(
                    {
                        "epoch": global_epoch,
                        "batch": global_batch,
                        "lr": model.optimizer_regime.get_lr()[0],
                    }
                )

            # Logging
            if batch % log_interval == 0 or batch == len(loader):
                logging.debug(
                    "{0}: epoch: {1} [{2}/{3}]\t".format(
                        prefix,
                        epoch + 1,
                        batch % len(loader) + 1,
                        len(loader),
                    )
                )
                for key, value in meters.items():
                    logging.debug(f"{key} {value.avg}\t")

                if measure_performance(dict(task_id=task_id, epoch=epoch, batch=batch)):
                    batch_metrics = model.batch_metrics.get(
                        dict(task_id=task_id, epoch=epoch, batch=batch)
                    )
                    logging.debug(f"batch {batch} time {last_batch_time} sec")
                    logging.debug(
                        f"\t[Python] batch load time {batch_metrics.get('load_time', -1)} sec ({batch_metrics.get('load_time', -1)*100/last_batch_time}%)"
                    )
                    logging.debug(
                        f"\t[Python] batch train time {batch_metrics.get('train_time', -1)} sec ({batch_metrics.get('train_time', -1)*100/last_batch_time}%)"
                    )
                    logging.debug(
                        f"\t[Python] batch accumulate time {batch_metrics.get('accumulate_time', -1)} sec ({batch_metrics.get('accumulate_time', -1)*100/last_batch_time}%)"
                    )

                    if model.use_memory_buffer:
                        logging.debug(
                            f"\t[Python] batch wait time {batch_metrics.get('wait_time', -1)} sec ({batch_metrics.get('wait_time', -1)*100/last_batch_time}%)"
                        )
                        logging.debug(
                            f"\t[Python] batch assemble time {batch_metrics.get('assemble_time', -1)} sec ({batch_metrics.get('assemble_time', -1)*100/last_batch_time}%)"
                        )
                        logging.debug(
                            f"\t[C++] batch copy time {batch_metrics.get('batch_copy_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] bulk prepare time {batch_metrics.get('bulk_prepare_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] rpcs resolve time {batch_metrics.get('rpcs_resolve_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] representatives copy time {batch_metrics.get('representatives_copy_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] buffer update time {batch_metrics.get('buffer_update_time', -1)} sec"
                        )
                        logging.debug(
                            f"\t[C++] local_rehearsal_size {meters['local_rehearsal_size'].val.item()}"
                        )

            avg_meters = {}
            for key, value in meters.items():
                avg_meters[key] = value.avg.item()
            progress.set_postfix(avg_meters)
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

    avg_meters = {}
    for key, value in meters.items():
        avg_meters[key] = value.avg.item()
    avg_meters["time"] = epoch_time
    avg_meters["batch"] = batch

    logging.info(f"\nCUMULATED VALUES:")
    logging.info(f"\tlocal_rehearsal_size {avg_meters['local_rehearsal_size']}")
    logging.info(f"epoch time {epoch_time} sec")

    return avg_meters


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
    # In case of an in-memory dataset that does not fit entirely in memory,
    # num_tasks should be defined however len(tasksets) will be wrong.
    num_tasks = train_data_regime.get("tasks").get("num_tasks") or len(
        train_data_regime.tasksets
    )
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

    # meters contains metrics for one epoch (average over all mini-batches)
    metrics_to_average = (
        ["loss", "prec1", "prec5"]
        if validate_data_regime.get("tasks").get("scenario", None) != "reconstruction"
        else ["loss", "loss_amp", "loss_ph"]
    )

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

            loader = train_data_regime.get_loader(task_id)

            # Horovod: set epoch to sampler for shuffling
            train_data_regime.set_epoch(epoch)
            train_results = train_one_epoch(
                model,
                loader,
                task_id,
                epoch,
                global_batch,
                global_epoch,
                train_data_regime.get("tasks").get("scenario", None),
                log_interval=log_interval,
            )

            if hvd.rank() == 0:
                meters = {}
                for key, value in train_results.items():
                    meters[f"train_{key}"] = train_results[key]
                wandb.log(
                    {
                        "epoch": global_epoch,
                        **meters,
                    }
                )

            # evaluate on test set
            before_evaluate_time = time.perf_counter()
            tasks_meters = []
            if evaluate or epoch + 1 == epochs[task_id]:
                for test_task_id in range(0, task_id + 1):
                    logging.info(
                        f"EVALUATING on task {test_task_id + 1}..{task_id + 1}"
                    )

                    tasks_meters.append({})

                    validate_data_regime.set_epoch(epoch)
                    validate_results = evaluate_one_epoch(
                        model,
                        validate_data_regime.get_loader(test_task_id),
                        task_id,
                        test_task_id,
                        epoch,
                        validate_data_regime.get("tasks").get("scenario", None),
                    )

                    if hvd.rank() == 0 and test_task_id == task_id:
                        meters = {}
                        for key, value in validate_results.items():
                            meters[f"val_{key}"] = validate_results[key]
                        wandb.log(
                            {
                                "epoch": global_epoch,
                                **meters,
                            }
                        )

                    if hvd.rank() == 0:
                        logging.info(
                            "RESULTS: Validate loss: {validate[loss]:.4f}\n".format(
                                validate=validate_results
                            )
                        )

                    for key, value in validate_results.items():
                        tasks_meters[test_task_id][key] = value
                    task_metrics["test_tasks_metrics"].append(
                        {
                            "epoch": epoch,
                            "test_task_id": test_task_id,
                            **tasks_meters[test_task_id],
                        }
                    )

                averages = {
                    metric: sum(tasks_meters[i][metric] for i in range(task_id + 1))
                    / (task_id + 1)
                    for metric in metrics_to_average
                }
                task_metrics["task_averages"].append(
                    {
                        "epoch": epoch,
                        **averages,
                    }
                )

                if hvd.rank() == 0:
                    meters = {}
                    for key in metrics_to_average:
                        meters[f"continual_task1_val_{key}"] = tasks_meters[0][key]
                        meters[f"continual_val_{key}"] = averages[key]
                    wandb.log(
                        {
                            "epoch": global_epoch,
                            **meters,
                        }
                    )

                # TODO: maybe we should compare the averaged loss on all previous tasks?
                if tasks_meters[task_id]["loss"] < model.minimal_eval_loss:
                    logging.debug(
                        f"Saving best model with minimal eval loss ({tasks_meters[task_id]['loss']}).."
                    )
                    model.minimal_eval_loss = tasks_meters[task_id]["loss"]
                    model.best_model = copy.deepcopy(model.backbone.state_dict())

            evaluate_durations.append(time.perf_counter() - before_evaluate_time)

            global_epoch += 1
            global_batch += train_results["batch"]

            if hvd.rank() == 0:
                img_sec = (
                    len(loader) * train_results["num_samples"] / train_results["time"]
                )
                img_secs.append(img_sec)
                logging.info(
                    "\nRESULTS: Time taken for epoch {} on {} device(s) is {} sec\n"
                    "Average: {} samples/sec per device\n"
                    "Average on {} device(s): {} samples/sec\n".format(
                        epoch + 1,
                        hvd.size(),
                        train_results["time"],
                        img_sec,
                        hvd.size(),
                        img_sec * hvd.size(),
                    )
                )
                for key in metrics_to_average:
                    logging.info(
                        f"Averaged eval {key} on all previous tasks: {averages[key]}"
                    )
                logging.info(
                    "Training loss: {train[loss]:.4f}\n".format(
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
            "Average on %d device(s): %.1f +-%.1f samples/sec\n"
            % (
                hvd.size(),
                hvd.size() * img_sec_mean,
                hvd.size() * img_sec_conf,
            )
        )
        for key in metrics_to_average:
            logging.info(f"Averaged eval {key} on all previous tasks: {averages[key]}")

        values = {
            "total_time": total_time,
            "total_train_time": total_training_time,
            "train_img_sec": img_sec_mean,
            "train_total_img_sec": img_sec_mean * hvd.size(),
        }
        if time_metrics is not None:
            time_metrics.add(**values)
            time_metrics.save()
