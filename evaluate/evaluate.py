import horovod.torch as hvd
import logging
import time
import torch
import wandb

from tqdm import tqdm
from torch.cuda.amp import autocast

from utils.meters import AverageMeter
from utils.log import get_logging_level


def evaluate_one_epoch(model, loader, task_id, test_task_id, epoch, scenario):
    model.backbone.eval()

    with torch.no_grad():
        prefix = "val"
        metrics = (
            ["loss", "prec1", "prec5", "num_samples", "local_rehearsal_size"]
            if scenario != "reconstruction"
            else ["loss", "loss_amp", "loss_ph", "num_samples", "local_rehearsal_size"]
        )
        meters = {metric: AverageMeter(f"{prefix}_{metric}") for metric in metrics}
        batch = 0
        epoch_time = 0
        last_batch_time = 0

        enable_tqdm = get_logging_level() >= logging.INFO and hvd.rank() == 0
        with tqdm(
            total=len(loader),
            desc=f"Task #{task_id + 1} {prefix} epoch #{epoch + 1}",
            disable=not enable_tqdm,
        ) as progress:
            start_batch_time = time.perf_counter()

            for data in loader:
                evaluate_fn = (
                    model.evaluate_one_step
                    if scenario != "reconstruction"
                    else model.evaluate_recon_one_step
                )
                evaluate_fn(data, meters)

                last_batch_time = time.perf_counter() - start_batch_time
                epoch_time += last_batch_time

                avg_meters = {}
                for key, value in meters.items():
                    avg_meters[key] = value.avg.item()
                progress.set_postfix(avg_meters)
                progress.update(1)

                batch += 1

        avg_meters = {}
        for key, value in meters.items():
            avg_meters[key] = value.avg.item()
        avg_meters["time"] = epoch_time
        avg_meters["batch"] = batch

        logging.info("epoch time %d sec", epoch_time)

        return avg_meters
