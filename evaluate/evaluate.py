def validate_one_epoch(self, data_regime, task_id):
        previous_task = task_id != self.task_id

        prefix = "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5", "num_samples"]
        }
        epoch_time = 0
        last_batch_time = 0
        loader = data_regime.get_loader(task_id)

        criterion = torch.nn.CrossEntropyLoss()

        enable_tqdm = self.log_level in ('info') and hvd.rank() == 0
        with tqdm(total=len(loader),
              desc=f"Task #{self.task_id + 1} {prefix} epoch #{self.epoch}",
              disable=not enable_tqdm
        ) as progress:
            start_batch_time = time.perf_counter()

            for x, y, _ in loader:
                x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)

                if hvd.rank() == 0 and False:
                    captions = []
                    for label in y:
                        captions.append(f"y={label.item()}")
                    display(f"val_batch_{self.task_id}_{task_id}_{self.epoch}_{self.batch}", x, captions=captions)

                if self.use_amp:
                    with autocast():
                        output = self.model(x)
                        loss = criterion(output, y)
                else:
                    output = self.model(x)
                    loss = criterion(output, y)

                prec1, prec5 = accuracy(output, y, topk=(1, 5))
                meters["loss"].update(loss, x.size(0))
                meters["prec1"].update(prec1, x.size(0))
                meters["prec5"].update(prec5, x.size(0))

                last_batch_time = time.perf_counter() - start_batch_time
                epoch_time += last_batch_time

                progress.set_postfix({'loss': meters["loss"].avg.item(),
                           'accuracy': meters["prec1"].avg.item(),
                           'augmented': self.aug_size - self.batch_size,
                           'local_rehearsal_size': self.current_rehearsal_size})
                progress.update(1)

        if hvd.rank() == 0 and not previous_task:
            wandb.log({"epoch": self.global_epoch,
                    f"{prefix}_loss": meters["loss"].avg,
                    f"{prefix}_prec1": meters["prec1"].avg,
                    f"{prefix}_prec5": meters["prec5"].avg})

        logging.info(f"epoch time {epoch_time} sec")

        num_samples = meters["num_samples"].sum.item()
        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["num_samples"] = num_samples
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = epoch_time
        meters["batch"] = self.batch

        return meters