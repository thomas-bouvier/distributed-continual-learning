import torch
import torch.nn as nn
import torch.nn.functional as F

class icarl_resnet_model(nn.Module):
    def __init__(self, num_classes, nf, resnet_size = 18):
        super(icarl_resnet_model, self).__init__()

        params = getResNetParameters(resnet_size)
        block = params[0]
        num_blocks = params[1]

        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        fv = out
        out = self.linear(out)
        return fv, out


def icarl_resnet(**config):
    return icarl_resnet_model()
