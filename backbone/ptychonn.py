import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

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
            nn.BatchNorm2d(filters_out) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        ]
        return block

    def up_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(filters_out) if self.use_batch_norm else nn.Identity(),
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
    # Scaling by the global rank degraded the accuracy when doing rehearsal
    lr = config.pop("lr", 5e-4)
    lr_min = config.pop("lr_min", 1e-4)
    batch_size = config.pop("batch_size")
    step_cycle_size = config.pop("step_cycle_size", 2944) / batch_size / world_size
    lr_schedule = config.pop("lr_schedule", "exp_range_cyclic_lr")

    model = PtychoNNModel()

    # https://github.com/bckenstler/CLR
    def triangular_cyclic_lr(step, lr, max_lr, step_size):
        """
        step_size: number of GLOBAL batches to complete a cycle
        """
        lr_batch = step["global_batch"]
        cycle = np.floor(1 + lr_batch / (2 * step_size))
        x = np.abs(lr_batch / step_size - 2 * cycle + 1)
        return lr + (max_lr - lr) * np.maximum(0, (1 - x))

    # https://github.com/bckenstler/CLR
    def triangular2_cyclic_lr(step, lr, max_lr, step_size):
        """
        step_size: number of GLOBAL batches to complete a cycle
        """
        lr_batch = step["global_batch"]
        cycle = np.floor(1 + lr_batch / (2 * step_size))
        x = np.abs(lr_batch / step_size - 2 * cycle + 1)
        return lr + (max_lr - lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))

    # https://github.com/bckenstler/CLR
    def exp_range_cyclic_lr(step, lr, max_lr, step_size, gamma=0.99985):
        """
        step_size: number of GLOBAL batches to complete a cycle
        """
        lr_batch = step["global_batch"]
        cycle = np.floor(1 + lr_batch / (2 * step_size))
        x = np.abs(lr_batch / step_size - 2 * cycle + 1)
        return lr + (max_lr - lr) * np.maximum(0, (1 - x)) * gamma ** (lr_batch)

    schedules = {
        "triangular_cyclic_lr": triangular_cyclic_lr,
        "triangular2_cyclic_lr": triangular2_cyclic_lr,
        "exp_range_cyclic_lr": exp_range_cyclic_lr,
    }

    def config_by_step(step):
        return {"lr": schedules[lr_schedule](step, lr_min, lr, step_cycle_size)}

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "Adam",
            "step_lambda": config_by_step,
        }
    ]

    return model
