import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['resnet']

def conv3x3(in_channels, out_channels, stride=1, groups=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=bias)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.bn2.weight, 0)

    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()


def weight_decay_config(value=1e-4, log=False):
    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: not n.endswith('bias'),
                       'module': lambda m: not isinstance(m, nn.BatchNorm2d)}
            }


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1,
                 downsample=None, groups=1, residual_block=None, dropout=0.):
        super(BasicBlock, self).__init__()

        dropout = 0 if dropout is None else dropout
        self.conv1 = conv3x3(in_channels, out_channels, stride, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, expansion * out_channels, groups=groups)
        self.bn2 = nn.BatchNorm2d(expansion * out_channels)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4,
                 downsample=None, groups=1, residual_block=None, dropout=0.):
        super(Bottleneck, self).__init__()
        dropout = 0 if dropout is None else dropout
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class resnet_model(nn.Module):
    def __init__(self):
        super(resnet_model, self).__init__()
        self.feature_vector = None

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1,
                    groups=1, residual_block=None, dropout=None):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.in_channels != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.in_channels, planes, stride,
                            expansion=expansion, downsample=downsample,
                            groups=groups, residual_block=residual_block,
                            dropout=dropout))
        self.in_channels = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, planes, expansion=expansion,
                                groups=groups, residual_block=residual_block,
                                dropout=dropout))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        self.feature_vector = x
        x = self.fc(x)
        return x


class resnet_cifar_model(resnet_model):
    def __init__(self, num_classes=10, in_channels=16,
                 block=BasicBlock, residual_block=None, layers=[2, 2, 2],
                 groups=[1, 1, 1], dropout=None, **kwargs):
        super(resnet_cifar_model, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x

        width = [2**i * in_channels for i in range(3)]
        self.layer1 = self._make_layer(block=block, planes=width[0], blocks=layers[0], groups=groups[0],
                                       residual_block=residual_block, dropout=dropout)
        self.layer2 = self._make_layer(block=block, planes=width[1], blocks=layers[1], stride=2, groups=groups[1],
                                       residual_block=residual_block, dropout=dropout)
        self.layer3 = self._make_layer(block=block, planes=width[2], blocks=layers[2], stride=2, groups=groups[2],
                                       residual_block=residual_block, dropout=dropout)
        self.layer4 = lambda x: x
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width[-1], num_classes)
        self.num_features = width[-1] * expansion

        init_model(self)
        self.regime = [{
                'epoch': 0,
                'optimizer': 'SGD',
                'lr': 1e-1,
                'momentum': 0.9,
                'regularizer': weight_decay_config(1e-4)
            },
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3},
            {'epoch': 164, 'lr': 1e-4}
        ]


class resnet_imagenet_model(resnet_model):
    def __init__(self, num_classes=1000, in_channels=64,
                 block=Bottleneck, residual_block=None, layers=[3, 4, 23, 3],
                 expansion=4, groups=[1, 1, 1, 1], lr=1, **kwargs):
        super(resnet_imagenet_model, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        width = [2**i * in_channels for i in range(len(layers))]
        for i in range(len(layers)):
            layer = self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion,
                                     stride=1 if i == 0 else 2, residual_block=residual_block, groups=groups[i])
            setattr(self, 'layer%s' % str(i + 1), layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width[-1] * expansion, num_classes)
        self.num_features = width[-1] * expansion

        init_model(self)
        self.regime = [{
                'epoch': 0,
                'optimizer': 'SGD',
                'lr': lr,
                'lr_rampup': True,
                'momentum': 0.9,
                'regularizer': weight_decay_config(1e-5)
            }, {
                'epoch': 5,
                'lr': lr * 1.,
                'lr_rampup': False
            },
            {'epoch': 30, 'lr': lr * 1e-1},
            {'epoch': 45, 'lr': lr * 1e-2},
            {'epoch': 80, 'lr': lr * 1e-3}
        ]


def resnet(config):
    dataset = config.get('dataset')
    depth = config.get('depth', 50)
    if depth == 18:
        config.update(dict(block=BasicBlock, in_channels=20))
    elif depth == 34:
        config.update(dict(block=BasicBlock, in_channels=40))
    elif depth == 50:
        config.update(dict(block=Bottleneck, in_channels=64))
    elif depth == 101:
        config.update(dict(block=Bottleneck, in_channels=64))
    elif depth == 152:
        config.update(dict(block=Bottleneck, in_channels=64))

    if dataset == 'cifar10':
        return resnet_cifar_model(num_classes=10, **config)

    elif dataset == 'cifar100':
        return resnet_cifar_model(num_classes=100, **config)

    elif 'imagenet' in dataset:
        if depth == 18:
            config.update(dict(layers=[2, 2, 2, 2], expansion=1))
        elif depth == 34:
            config.update(dict(layers=[3, 4, 6, 3], expansion=1))
        elif depth == 50:
            config.update(dict(layers=[3, 4, 6, 3]))
        elif depth == 101:
            config.update(dict(layers=[3, 4, 23, 3]))
        elif depth == 152:
            config.update(dict(layers=[3, 8, 36, 3]))
        if dataset == 'imagenet100':
            num_classes = 100
        elif dataset == 'tinyimagenet':
            num_classes = 200
        else:
            num_classes = 1000
        return resnet_imagenet_model(num_classes=num_classes, **config)

    elif dataset == 'core50':
        if depth == 18:
            config.update(dict(layers=[2, 2, 2, 2], expansion=1))
        elif depth == 34:
            config.update(dict(layers=[3, 4, 6, 3], expansion=1))
        elif depth == 50:
            config.update(dict(layers=[3, 4, 6, 3]))
        elif depth == 101:
            config.update(dict(layers=[3, 4, 23, 3]))
        elif depth == 152:
            config.update(dict(layers=[3, 8, 36, 3]))
        return resnet_imagenet_model(num_classes=50, **config)

    else:
        raise ValueError('Unknown dataset')
