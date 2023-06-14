"""Forward pass for the current epoch"""
def train_one_epoch(model, data_regime):
    model.train()

    prefix = "train"
    meters = {
        metric: AverageMeter(f"{prefix}_{metric}")
        for metric in ["loss", "prec1", "prec5", "num_samples"]
    }
    epoch_time = 0
    last_batch_time = 0
    loader = data_regime.get_loader(self.task_id)

    enable_tqdm = self.log_level in ('info') and hvd.rank() == 0
    with tqdm(total=len(loader),
            desc=f"Task #{self.task_id + 1} {prefix} epoch #{self.epoch + 1}",
            disable=not enable_tqdm
    ) as progress:
        timer = self.get_timer('load', previous_iteration=model.use_memory_buffer)
        timer.__enter__()
        start_batch_time = time.perf_counter()

        for x, y, _ in loader:
            x, y = move_cuda(x, self.cuda), move_cuda(y.long(), self.cuda)
            timer.__exit__(None, None, None)

            model.train_one_step(x, y, meters)

            if hvd.rank() == 0 and False:
                captions = []
                for label in y:
                    captions.append(f"y={label.item()}")
                display(f"train_batch_{self.task_id}_{self.epoch}_{self.batch}", x, captions=captions)

            last_batch_time = time.perf_counter() - start_batch_time
            epoch_time += last_batch_time

            iteration = self.batch
            if model.use_memory_buffer:
                iteration -= 1

            if hvd.rank() == 0:
                # Performance metrics
                if self.measure_performance() and self.batch_metrics is not None:
                    metrics = self.perf_metrics.get(iteration)

                    batch_metrics_values = dict(
                        epoch=self.epoch,
                        batch=self.batch,
                        time=last_batch_time,
                        load_time=metrics.get('load', 0),
                        train_time=metrics.get('train', 0),
                    )
                    if model.use_memory_buffer:
                        batch_metrics_values |= dict(
                            wait_time=metrics.get('wait', 0),
                            assemble_time=metrics.get('assemble', 0),
                            accumulate_time=metrics[0],
                            copy_time=metrics[1],
                            bulk_prepare_time=metrics[2],
                            rpcs_resolve_time=metrics[3],
                            representatives_copy_time=metrics[4],
                            buffer_update_time=metrics[5],
                            aug_size=self.aug_size,
                            local_rehearsal_size=self.current_rehearsal_size,
                        )

                    self.batch_metrics.add(**batch_metrics_values)
                    self.batch_metrics.save()

            # Logging
            if self.batch % self.log_interval == 0 or self.batch == len(loader):
                logging.debug(
                    "{0}: epoch: {1} [{2}/{3}]\t"
                    "Loss {meters[loss].avg:.4f}\t"
                    "Prec@1 {meters[prec1].avg:.3f}\t"
                    "Prec@5 {meters[prec5].avg:.3f}\t".format(
                        prefix,
                        self.epoch + 1,
                        self.batch % len(loader) + 1,
                        len(loader),
                        meters=meters,
                    )
                )

                if self.measure_performance():
                    metrics = self.perf_metrics.get(iteration)
                    logging.debug(f"batch {self.batch} time {last_batch_time} sec")
                    logging.debug(
                        f"\t[Python] batch load time {metrics.get('load', 0)} sec ({metrics.get('load', 0)*100/last_batch_time}%)")
                    logging.debug(
                        f"\t[Python] batch train time {metrics.get('train', 0)} sec ({metrics.get('train', 0)*100/last_batch_time}%)")

                    if model.use_memory_buffer:
                        logging.debug(
                            f"\t[Python] batch wait time {metrics.get('wait', 0)} sec ({metrics.get('wait', 0)*100/last_batch_time}%)")
                        logging.debug(
                            f"\t[Python] batch assemble time {metrics.get('assemble', 0)} sec ({metrics.get('assemble', 0)*100/last_batch_time}%)")
                        logging.debug(
                            f"\t[C++] batch accumulate time {metrics[0]} sec")
                        logging.debug(
                            f"\t[C++] batch copy time {metrics[1]} sec")
                        logging.debug(
                            f"\t[C++] bulk prepare time {metrics[2]} sec")
                        logging.debug(
                            f"\t[C++] rpcs resolve time {metrics[3]} sec")
                        logging.debug(
                            f"\t[C++] representatives copy time {metrics[4]} sec")
                        logging.debug(
                            f"\t[C++] buffer update time {metrics[5]} sec")
                        logging.debug(
                            f"\t[C++] local_rehearsal_size {self.current_rehearsal_size}")

            tqdm_postfix = dict(
                loss=meters["loss"].avg.item(),
                accuracy=meters["prec1"].avg.item(),
            )
            progress.set_postfix({'loss': meters["loss"].avg.item(),
                        'accuracy': meters["prec1"].avg.item(),
                        'augmented': self.aug_size - self.batch_size,
                        'local_rehearsal_size': self.current_rehearsal_size})
            progress.update(1)

            self.global_batch += 1
            self.batch += 1

            timer = self.get_timer('load', previous_iteration=model.use_memory_buffer)
            timer.__enter__()
            start_batch_time = time.perf_counter()

    if hvd.rank() == 0:
        wandb.log({"epoch": self.global_epoch,
                f"{prefix}_loss": meters["loss"].avg,
                f"{prefix}_prec1": meters["prec1"].avg,
                f"{prefix}_prec5": meters["prec5"].avg,
                "lr": self.optimizer_regime.get_lr()[0],
                "local_rehearsal_size": self.current_rehearsal_size})

    self.global_epoch += 1
    self.epoch += 1

    logging.info(f"\nCUMULATED VALUES:")
    logging.info(
        f"\tlocal_rehearsal_size {self.current_rehearsal_size}")
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

    num_samples = meters["num_samples"].sum.item()
    meters = {name: meter.avg.item() for name, meter in meters.items()}
    meters["num_samples"] = num_samples
    meters["error1"] = 100.0 - meters["prec1"]
    meters["error5"] = 100.0 - meters["prec5"]
    meters["time"] = epoch_time
    meters["batch"] = self.batch

    return meters