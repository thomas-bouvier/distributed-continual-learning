import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ptychonn"]


class PtychoNNModel(nn.Module):
    def __init__(self, nconv: int = 16):
        super(PtychoNN, self).__init__()
        self.nconv = nconv

        self.encoder = nn.Sequential(  # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
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

    def down_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(in_channels=filters_in, out_channels=filters_out,
                      kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        ]
        return block

    def up_block(self, filters_in, filters_out):
        block = [
            nn.Conv2d(filters_in, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, 3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ]
        return block

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x1 = self.encoder(x)
            #amp = self.decoder1(x1)
            ph = self.decoder2(x1)

            # Restore -pi to pi range
            # Using tanh activation (-1 to 1) for phase so multiply by pi
            ph = ph*np.pi

        return ph


def ptychonn(config):
    self.iters_per_epoch = len(self.data_regime.get_train_loader())
    self.epochs_per_half_cycle = epochs_per_half_cycle
    self.iters_per_half_cycle = epochs_per_half_cycle * \
        self.iters_per_epoch  # Paper recommends 2-10 number of iterations

    logging.info(
        f"LR step size is: {self.iters_per_half_cycle} which is every {self.iters_per_half_cycle / self.iters_per_epoch} epochs")

    self.max_lr = max_lr
    self.min_lr = min_lr

    #criterion = lambda t1, t2: nn.L1Loss()
    self.criterion = self.customLoss
    optimizer = torch.optim.Adam(
        self.model.parameters(), lr=self.max_lr)
    self.scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=self.max_lr, base_lr=self.min_lr,
                                                        step_size_up=self.iters_per_half_cycle,
                                                        cycle_momentum=False, mode='triangular2')


    return PtychoNNModel()