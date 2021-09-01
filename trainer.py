from utils.timer import Timer

import torch.nn as nn
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, model, log_interval, optimizer=None):
        self.timer = Timer()

        self.model = model
        self.log_interval = log_interval
        self.epoch = 0
        self.training_steps = 0
        self.optimizer = optimizer


    def train(self, data_loader):
        # switch to train mode
        self.model.train()
        self.forward(data_loader)
    

    def forward(self, data_loader):
        for batch_idx, (input, target) in enumerate(data_loader):
            self.timer.start(f"start_epoch_{self.epoch}-batch_{batch_idx}")
            self.timer.start(f"epoch_{self.epoch }-batch_{batch_idx}")

            if self.model.cuda:
                self.timer.start(f"epoch_{self.epoch }-batch_{batch_idx}-move_batch_to_gpu")
                input, target = input.cuda(), target.cuda()
                self.timer.end(f"epoch_{self.epoch }-batch_{batch_idx}-move_batch_to_gpu")

            self.timer.start(f"epoch_{self.epoch}-batch_{batch_idx}-zero_grad")
            self.optimizer.zero_grad()
            self.timer.end(f"epoch_{self.epoch }-batch_{batch_idx}-zero_grad")

            self.timer.start(f"epoch_{self.epoch }-batch_{batch_idx}-forward_pass")
            output = self.model(input)
            self.timer.end(f"epoch_{self.epoch }-batch_{batch_idx}-forward_pass")

            self.timer.start(f"epoch_{self.epoch }-batch_{batch_idx}-compute_loss")
            loss = F.nll_loss(output, target)
            self.timer.end(f"epoch_{self.epoch }-batch_{batch_idx}-compute_loss")

            self.timer.start(f"epoch_{self.epoch }-batch_{batch_idx}-backward_pass")
            loss.backward()
            self.timer.end(f"epoch_{self.epoch }-batch_{batch_idx}-backward_pass")

            self.timer.start(f"epoch_{self.epoch }-batch_{batch_idx}-optimizer_step")
            self.optimizer.step()
            self.timer.end(f"epoch_{self.epoch }-batch_{batch_idx}-optimizer_step")

            self.timer.end(f"epoch_{self.epoch }-batch_{batch_idx}")
            self.timer.start(f"end_epoch_{self.epoch}-batch_{batch_idx}")

            if batch_idx % self.log_interval == 0 or batch_idx == len(data_loader) - 1:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print('Train Epoch: {} [{}/{}\tLoss: {:.6f}'.format(self.epoch, batch_idx, len(data_loader), loss.item()))

        return self.timer.retrieve()