import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size,
                              out_channels=out_channels, stride=stride,
                              padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


class FirstNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FirstNet, self).__init__()

        self.unit1 = Unit(in_channels=3, out_channels=32, kernel_size=5)
        self.unit2 = Unit(in_channels=32, out_channels=32, kernel_size=4)
        # self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        # self.unit5 = Unit(in_channels=64, out_channels=64)
        # self.unit6 = Unit(in_channels=64, out_channels=64)
        # self.unit7 = Unit(in_channels=64, out_channels=64)

        self.avgpool = nn.AvgPool2d(kernel_size=6)

        self.net = nn.Sequential(self.unit1, self.unit2,
                                 self.pool1, self.unit4,
                                 self.avgpool)

        self.fc = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        # print(output.shape)
        output = output.view(-1, 256)
        # print(output.shape)
        output = self.fc(output)
        return output


class SecondNet(nn.Module):
    def __init__(self):
        super(SecondNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AdvNet(nn.Module):
    def __init__(self, input_dim, activation_function=F.relu):
        super(AdvNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 10)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.linear1(x))
        x = self.activation_function(self.linear2(x))
        x = self.linear3(x)
        return x


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))
