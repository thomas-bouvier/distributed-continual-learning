from utils.timer import Timer

import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, model, optimizer, criterion, log_interval):
        self.timer = Timer()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.log_interval = log_interval
        self.epoch = 0
        self.training_steps = 0

    
    def _step(self, i_batch, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1):
        outputs = []
        total_loss = 0

        self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}")

        if training:
            self.timer.start(f"epoch_{self.epoch}-batch_{i_batch}-zero_grad")
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)
            self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-zero_grad")

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            if self.model.cuda:
                self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-move_batch_to_gpu")
                inputs, target = inputs.cuda(), target.cuda()
                self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-move_batch_to_gpu")

            self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-forward_pass")
            output = self.model(inputs)
            self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-forward_pass")

            self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-compute_loss")
            loss = self.criterion(output, target)
            self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-compute_loss")

            if training:
                self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-backward_pass")
                # accumulate gradient
                loss.backward()
                self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-backward_pass")

            if training:
                self.timer.start(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-optimizer_step")
                # SGD step
                self.optimizer.step()
                self.training_steps += 1
                self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}-chunk_{i}-optimizer_step")

            outputs.append(output.detach())
            total_loss += float(loss)

        self.timer.end(f"epoch_{self.epoch }-batch_{i_batch}")

        return outputs, total_loss

    def forward(self, data_loader, training=False, average_output=False):
        for i_batch, (inputs, target) in enumerate(data_loader):
            self.timer.start(f"start_epoch_{self.epoch}-batch_{i_batch}")
            output, loss = self._step(i_batch,
                                      inputs,
                                      target,
                                      training=training,
                                      average_output=average_output)
            self.timer.start(f"end_epoch_{self.epoch}-batch_{i_batch}")

            if i_batch % self.log_interval == 0 or i_batch == len(data_loader) - 1:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{}] \tLoss: {:.6f}'.format(self.epoch, i_batch, len(data_loader), loss))

        return self.timer.retrieve()


    def train(self, data_loader, average_output=False):
        # switch to train mode
        self.model.train()
        self.forward(data_loader, average_output=average_output, training=True)


    def validate(self, data_loader, average_output=False):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.forward(data_loader, average_output=average_output, training=False)