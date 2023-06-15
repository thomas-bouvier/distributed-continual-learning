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


self.agent.before_all_tasks(self.train_data_regime)

        for task_id in range(self.resume_from_task, len(self.train_data_regime.tasksets)):
            start = time.perf_counter()
            training_time = time.perf_counter()

            task_metrics = {"task_id": task_id, "test_tasks_metrics": [], "task_averages": []}

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
                        self.validate_data_regime.set_epoch(i_epoch)

                        logging.info(f"EVALUATING on task {test_task_id + 1}..{task_id + 1}")

                        validate_results = self.agent.validate(self.validate_data_regime, test_task_id)
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