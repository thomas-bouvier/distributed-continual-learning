import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cross_entropy import ScaledMeanAbsoluteErrorLoss

__all__ = ["ptychonn"]


class PtychoNNModel(nn.Module):
    def __init__(self, nconv=32, use_batch_norm=False):
        super().__init__()
        self.nconv = nconv
        self.use_batch_norm = use_batch_norm

        self.encoder = nn.Sequential(
            *self.down_block(1, self.nconv),
            *self.down_block(self.nconv, self.nconv * 2),
            *self.down_block(self.nconv * 2, self.nconv * 4),
            *self.down_block(self.nconv * 4, self.nconv * 8),
        )

        # amplitude model
        self.decoder1 = nn.Sequential(
            *self.up_block(self.nconv * 8, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 4),
            *self.up_block(self.nconv * 4, self.nconv * 2),
            *self.up_block(self.nconv * 2, self.nconv * 1),
            nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1, 1)),
        )

        # phase model
        self.decoder2 = nn.Sequential(
            *self.up_block(self.nconv * 8, self.nconv * 8),
            *self.up_block(self.nconv * 8, self.nconv * 4),
            *self.up_block(self.nconv * 4, self.nconv * 2),
            *self.up_block(self.nconv * 2, self.nconv * 1),
            nn.Conv2d(self.nconv * 1, 1, 3, stride=1, padding=(1, 1)),
            nn.Tanh(),
        )

        # self.criterion = nn.L1Loss(reduction="none")
        self.criterion = ScaledMeanAbsoluteErrorLoss(scaling=1.0)

    def down_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(
                in_channels=filters_in,
                out_channels=filters_out,
                kernel_size=3,
                stride=1,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(filters_out) if self.use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out) if self.use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        ]
        return block

    def up_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out) if self.use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out) if self.use_batch_norm else torch.nn.Identity(),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        ]
        return block

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            amp = self.decoder1(x1)
            ph = self.decoder2(x1)

            # Restore -pi to pi range
            # Using tanh activation (-1 to 1) for phase so multiply by pi
            ph = ph * np.pi

        return amp, ph


def ptychonn(config):
    world_size = config.pop("world_size", 1)
    lr = config.pop("lr", 5e-4) * world_size
    lr_min = config.pop("lr_min", 1e-4) * world_size
    warmup_epochs = config.pop("warmup_epochs", 0)
    num_steps_per_epoch = config.pop("num_steps_per_epoch")
    num_epochs = config.pop("num_epochs")
    num_samples = config.pop("total_num_samples") / world_size

    model = PtychoNNModel()

    # https://github.com/bckenstler/CLR
    def triangular2_cyclic_lr(step, lr, max_lr, step_size):
        """
        step_size: number of GLOBAL epochs to complete a cycle
        """
        lr_epoch = (
            step["task_id"] * num_epochs
            + step["epoch"]
            + step["batch"] / num_steps_per_epoch
        )
        cycle = np.floor(1 + lr_epoch / (2 * step_size))
        x = np.abs(lr_epoch / step_size - 2 * cycle + 1)
        return lr + (max_lr - lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))

    # https://github.com/bckenstler/CLR
    def exp_range_cyclic_lr(step, lr, max_lr, step_size, gamma=0.992):
        """
        step_size: number of GLOBAL epochs to complete a cycle
        """
        lr_epoch = (
            step["task_id"] * num_epochs
            + step["epoch"]
            + step["batch"] / num_steps_per_epoch
        )
        cycle = np.floor(1 + lr_epoch / (2 * step_size))
        x = np.abs(lr_epoch / step_size - 2 * cycle + 1)
        return lr + (max_lr - lr) * np.maximum(0, (1 - x)) * gamma ** (lr_epoch)

    def config_by_step(step):
        step_size = 16
        return {"lr": exp_range_cyclic_lr(step, lr_min, lr, step_size)}

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "Adam",
            "step_lambda": config_by_step,
        }
    ]

    return model
