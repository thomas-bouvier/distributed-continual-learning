import horovod.torch as hvd
import math
import torch.nn as nn
import logging
import timm

__all__ = ["efficientnetv2"]

def modify_drop_connect_rate(model, value, log=True):
    for m in model.modules():
        if hasattr(m, 'drop_prob'):
            if log and m.drop_prob != value:
                logging.debug('Modified drop-path rate from %s to %s' %
                              (m.drop_prob, value))
            m.drop_prob = value

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
    lr = config.pop("lr") * hvd.size()
    warmup_epochs = config.pop("warmup_epochs")
    num_epochs = config.pop("num_epochs")
    num_steps_per_epoch = config.pop("num_steps_per_epoch")

    # passing num_classes
    model = timm.create_model('efficientnetv2_m', pretrained=False, **config)

    def increase_drop_connect(epoch, drop_connect_rate=0.2):
        return lambda: modify_drop_connect_rate(model, min(drop_connect_rate, drop_connect_rate * epoch / float(num_epochs)))

    def rampup_lr(lr, step, num_steps_per_epoch, warmup_epochs):
        return lr * step / (num_steps_per_epoch * warmup_epochs)

    def config_by_step(step):
        warmup_steps = warmup_epochs * num_steps_per_epoch

        if step < warmup_steps:
            return {'lr': 1e-6 * hvd.size()}
            #return {'lr': rampup_lr(lr, step, num_steps_per_epoch, warmup_epochs)}
        return {}

    def config_by_epoch(epoch):
        return {'lr': lr * (0.97 ** round(epoch/2.4)),
                'execute': increase_drop_connect(epoch)}

    """RMSProp optimizer with
    decay 0.9 and momentum 0.9;
    weight decay 1e-5; initial learning rate 0.256 that decays
    by 0.97 every 2.4 epochs"""
    model.regime = [
        {
            "epoch": 0,
            'optimizer': 'RMSprop',
            'alpha': 0.9,
            'momentum': 0.9,
            "weight_decay": 0.00001,
            "step_lambda": config_by_step,
        },
        {
            'epoch': warmup_epochs,
            'regularizer': weight_decay_config(1e-5),
            'epoch_lambda': config_by_epoch
        }
    ]

    return model
