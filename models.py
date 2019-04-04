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
