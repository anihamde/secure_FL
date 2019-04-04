import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size, out_channels=out_channels, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output
 
class SimpleNet(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleNet,self).__init__()
        
        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3, out_channels=32, kernel_size=5)
        self.unit2 = Unit(in_channels=32, out_channels=32, kernel_size=4)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, 
                                self.unit4, self.unit5, self.unit6, self.unit7, 
                                self.avgpool)

        self.fc = nn.Linear(in_features=64,out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,64)
        output = self.fc(output)

        return output
