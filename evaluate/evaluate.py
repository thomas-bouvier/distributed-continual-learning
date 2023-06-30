import horovod.torch as hvd
import logging
import time
import torch
import torch.nn as nn
import wandb

from tqdm import tqdm
from torch.cuda.amp import autocast

from utils.meters import AverageMeter, accuracy
from utils.log import get_logging_level


def evaluate_one_epoch(model, loader, task_id, test_task_id, epoch):
    device = model._device()
    model.eval()

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

        criterion = torch.nn.CrossEntropyLoss()

        enable_tqdm = get_logging_level() in ('info') and hvd.rank() == 0
        with tqdm(total=len(loader),
              desc=f"Task #{task_id + 1} {prefix} epoch #{epoch}",
              disable=not enable_tqdm
        ) as progress:
            start_batch_time = time.perf_counter()

            for x, y, _ in loader:
                x, y = x.to(device), y.long().to(device)

                if hvd.rank() == 0 and False:
                    captions = []
                    for label in y:
                        captions.append(f"y={label.item()}")
                    display(f"val_batch_{task_id}_{epoch}_{batch}", x, captions=captions)

                with autocast(enabled=model.use_amp):
                    output = model.backbone(x)
                    loss = criterion(output, y)

                prec1, prec5 = accuracy(output, y, topk=(1, 5))
                meters["loss"].update(loss, x.size(0))
                meters["prec1"].update(prec1, x.size(0))
                meters["prec5"].update(prec5, x.size(0))

                last_batch_time = time.perf_counter() - start_batch_time
                epoch_time += last_batch_time

                progress.set_postfix({'loss': meters["loss"].avg.item(),
                           'accuracy': meters["prec1"].avg.item()})
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


"""
def validate(self):
    tot_val_loss = 0.0
    val_loss_ph = 0.0
    hvd_val_loss = Metric('val_loss')
    with tqdm(total=len(self.data_regime.get_validate_loader()),
                desc='Validate Epoch     #{}'.format(self.epoch + 1)) as bar:
        for ft_images, phs, t in self.data_regime.get_validate_loader():
            ft_images = ft_images.to(self.device)
            phs = phs.to(self.device)
            pred_phs = self.model(ft_images)

            val_loss_p = self.criterion(
                pred_phs, phs, len(self.data_regime.validate_data))
            val_loss = val_loss_p

            # try complex valued diff
            #diff_real = pred_amps * torch.cos(pred_phs) - amps * torch.cos(phs)
            #diff_imag = pred_amps * torch.sin(pred_phs) - amps * torch.sin(phs)
            #val_loss = torch.mean(torch.abs(diff_real + diff_imag))
            tot_val_loss += val_loss.detach().item()
            val_loss_ph += val_loss_p.detach().item()
            hvd_val_loss.update(val_loss_p)

            bar.set_postfix({'loss': val_loss.detach().item()})
            bar.update(1)

    self.metrics['val_losses'].append([tot_val_loss, hvd_val_loss.avg])

    self.saveMetrics(self.metrics, self.output_path, self.output_suffix)
    # Update saved model if val loss is lower

    if (tot_val_loss < self.metrics['best_val_loss']):
        logging.info(
            f"Saving improved model after Val Loss improved from {self.metrics['best_val_loss']} to {tot_val_loss}")
        self.metrics['best_val_loss'] = tot_val_loss
        self.updateSavedModel(
            self.model, self.output_path, self.output_suffix)
"""