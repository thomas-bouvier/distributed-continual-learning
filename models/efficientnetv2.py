import horovod.torch as hvd
import math
import torch.nn as nn
import logging
import timm

__all__ = ["efficientnetv2"]


def weight_decay_config(value=1e-4, log=False):
    def regularize_layer(m):
        non_depthwise_conv = isinstance(m, nn.Conv2d) \
            and m.groups != m.in_channels
        return not isinstance(m, nn.BatchNorm2d)

    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: not n.endswith('bias'),
                       'module': regularize_layer}
            }

def efficientnetv2(config):
    lr = config.pop("lr") * hvd.size() # 0.00075 * world_size
    lr_min = config.pop("lr_min") * hvd.size() # 1e-6 * world_size
    warmup_epochs = config.pop("warmup_epochs")
    num_epochs = config.pop("num_epochs")
    num_steps_per_epoch = config.pop("num_steps_per_epoch")

    # passing num_classes
    model = timm.create_model('efficientnetv2_m', pretrained=False, drop_rate=0.225, **config)

    def rampup_lr(lr, step, num_steps_per_epoch, warmup_epochs):
        return lr * step / (num_steps_per_epoch * warmup_epochs)

    def cosine_anneal_lr(lr, step, T_max, eta_min=1e-6):
        """
        Args:
            eta_min (float): lower lr bound for cyclic schedulers that hit 0 (1e-6)
        """
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * step / T_max)) / 2

    def config_by_step(step):
        warmup_steps = warmup_epochs * num_steps_per_epoch

        if step < warmup_steps:
            return {'lr': rampup_lr(lr, step, num_steps_per_epoch, warmup_epochs)}
        else:
            return {'lr': cosine_anneal_lr(lr, step, num_epochs * num_steps_per_epoch, eta_min=lr_min)}

    model.regime = [
        {
            "epoch": 0,
            'optimizer': 'RMSprop',
            'alpha': 0.9,
            'momentum': 0.9,
            'eps': 0.001,
            "weight_decay": 0.000015,
            "step_lambda": config_by_step,
        },
        {
            'epoch': warmup_epochs - 1,
            "step_lambda": config_by_step,
        }
    ]

    return model
