from modules import ContinualLearner, MemoryBuffer

class Classifier(ContinualLearner, MemoryBuffer):
    '''Model for classifying images, "enriched" as ContinualLearner- and MemoryBuffer-object.'''

    def __init__():
        super().__init__()
    

def train_one_step(
        self,
        x,
        y,
        meters,
    ):
        # Get the representatives
        with self.get_timer('wait', previous_iteration=True):
            self.aug_size = self.dsl.wait()
            n = self.aug_size - self.batch_size
            if n > 0:
                logging.debug(f"Received {n} samples from other nodes")

            if self.measure_performance():
                cpp_metrics = self.dsl.get_metrics(self.batch)
                self.perf_metrics.add(self.batch-1, cpp_metrics)

        # Assemble the minibatch
        with self.get_timer('assemble'):
            current_minibatch = self.get_current_augmented_minibatch()
            new_x = current_minibatch.x[:self.aug_size]
            new_y = current_minibatch.y[:self.aug_size]
            new_w = current_minibatch.w[:self.aug_size]

        if self.log_buffer and self.batch % self.log_interval == 0 and hvd.rank() == 0:
            repr_size = self.aug_size - self.batch_size
            if repr_size > 0:
                captions = []
                for label, weight in zip(new_y[-repr_size:], new_w[-repr_size:]):
                    captions.append(f"y={label.item()} w={weight.item()}")
                display(f"aug_batch_{self.task_id}_{self.epoch}_{self.batch}", new_x[-repr_size:], captions=captions)

        # In-advance preparation of next minibatch
        with self.get_timer('accumulate'):
            next_minibatch = self.get_next_augmented_minibatch()
            self.dsl.accumulate(x, y, next_minibatch.x, next_minibatch.y, next_minibatch.w)

        # Train
        with self.get_timer('train'):
            self.optimizer_regime.update(self.epoch, self.batch)
            self.optimizer_regime.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(new_x)
                    loss = self.criterion(output, new_y)
            else:
                output = self.model(new_x)
                loss = self.criterion(output, new_y)

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
            total_weight = hvd.allreduce(torch.sum(new_w), name='total_weight', op=hvd.Sum)
            dw = new_w / total_weight * self.batch_size * hvd.size() / self.batch_size

            if self.use_amp:
                self.scaler.scale(loss).backward(dw)
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward(dw)
                self.optimizer_regime.step()

            self.current_rehearsal_size = self.dsl.get_rehearsal_size()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, new_y, topk=(1, 5))
            meters["loss"].update(loss.sum() / self.mask[new_y].sum())
            meters["prec1"].update(prec1, new_x.size(0))
            meters["prec5"].update(prec5, new_x.size(0))
            meters["num_samples"].update(self.aug_size)