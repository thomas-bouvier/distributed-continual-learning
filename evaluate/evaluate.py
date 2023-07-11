import horovod.torch as hvd
import logging
import time
import torch
import wandb

from tqdm import tqdm
from torch.cuda.amp import autocast

from utils.meters import AverageMeter, accuracy
from utils.log import get_logging_level


def evaluate_one_epoch(model, loader, task_id, test_task_id, epoch):
    device = model._device()
    model.backbone.eval()

    with torch.no_grad():
        previous_task = test_task_id != task_id

        prefix = "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5"]
        }
        batch = 0
        epoch_time = 0
        last_batch_time = 0

        enable_tqdm = get_logging_level() in ("info") and hvd.rank() == 0
        with tqdm(
            total=len(loader),
            desc=f"Task #{task_id + 1} {prefix} epoch #{epoch}",
            disable=not enable_tqdm,
        ) as progress:
            start_batch_time = time.perf_counter()

            for x, y, _ in loader:
                x, y = x.to(device), y.long().to(device)

                step = dict(task_id=task_id, epoch=epoch, batch=batch)
                model.evaluate_one_step(x, y, meters, step)

                last_batch_time = time.perf_counter() - start_batch_time
                epoch_time += last_batch_time

                progress.set_postfix(
                    {
                        "loss": meters["loss"].avg.item(),
                        "accuracy": meters["prec1"].avg.item(),
                    }
                )
                progress.update(1)

                batch += 1

        meters["loss"] = meters["loss"].avg.item()
        meters["prec1"] = meters["prec1"].avg.item()
        meters["prec5"] = meters["prec5"].avg.item()
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = epoch_time
        meters["batch"] = batch

        logging.info(f"epoch time {epoch_time} sec")

        return meters
