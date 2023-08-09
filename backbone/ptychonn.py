import horovod.torch as hvd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cross_entropy import ScaledMeanAbsoluteErrorLoss

__all__ = ["ptychonn"]


class PtychoNNModel(nn.Module):
    def __init__(self, nconv: int = 16, loss_scaling=1.0):
        super().__init__()
        self.nconv = nconv

        self.encoder = nn.Sequential(
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8)
        )

        # amplitude model
        # self.decoder1 = nn.Sequential(
        #    *self.up_block(self.nconv * 8, self.nconv * 8),
        #    *self.up_block(self.nconv * 8, self.nconv * 4),
        #    *self.up_block(self.nconv * 4, self.nconv * 2),
        #    *self.up_block(self.nconv * 2, self.nconv * 1),
        #    nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1,1)),
        # )

        # phase model
        self.decoder2 = nn.Sequential(
            *self.up_block(self.nconv * 8, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 4),
            *self.up_block(self.nconv * 4, self.nconv * 2),
            *self.up_block(self.nconv * 2, self.nconv * 1),
            nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh()
        )

        self.criterion = ScaledMeanAbsoluteErrorLoss(scaling=loss_scaling)

    def down_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(
                in_channels=filters_in,
                out_channels=filters_out,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
            ),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        ]
        return block

    def up_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        ]
        return block

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            # amp = self.decoder1(x1)
            ph = self.decoder2(x1)

            # Restore -pi to pi range
            # Using tanh activation (-1 to 1) for phase so multiply by pi
            ph = ph * np.pi

        return ph


def ptychonn(config):
    lr = config.pop("lr", 5e-4) * hvd.size()
    lr_min = config.pop("lr_min", 1e-4) * hvd.size()
    warmup_epochs = config.pop("warmup_epochs")
    num_steps_per_epoch = config.pop("num_steps_per_epoch")
    num_samples = config.pop("total_num_samples") / hvd.size()

    model = PtychoNNModel(loss_scaling=num_samples)

    # https://github.com/bckenstler/CLR
    def triangular2_cyclic_lr(step, lr, max_lr, step_size):
        cycle = np.floor(1 + step["batch"] / (2 * step_size))
        x = np.abs(step["batch"] / step_size - 2 * cycle + 1)
        return lr + (max_lr - lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))

    def config_by_step(step):
        step_size = 6 * num_steps_per_epoch
        return {"lr": triangular2_cyclic_lr(step, lr_min, lr, step_size)}

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "Adam",
            "step_lambda": config_by_step,
        }
    ]

    return model
