import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torchvision
import numpy as np
from agents import *
from models import *

n_workers = 80
n_epochs = 100
batch_size = 16
mean0_std = 0  # 0 if no zero-mean epsilon
learning_rate = 0.001

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


model = FirstNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()


def rule(ups_list):  # ups_list is a tensor
    return mean(ups_list)[0]


central = Central(model, optimizer)
worker_list = []
for i in range(n_workers):
    worker_list.append(Worker(loss))
agg = Agg(rule)

e_dist_w = Normal(torch.zeros_like(central.model.weight), mean0_std)
e_dist_b = Normal(torch.zeros_like(central.model.bias), mean0_std)


for t in range(n_epochs):
    weight_ups = []
    bias_ups = []
    for i in range(n_workers):
        dataiter = iter(trainloader)
        batch_inp, batch_outp = dataiter.next()

        worker_list[i].model = central.model
        ups = worker_list[i].fwd_bkwd(batch_inp, batch_outp)
        ups[0] += e_dist_w.sample()
        ups[1] += e_dist_b.sample()
        weight_ups.append(ups[0])
        bias_ups.append(ups[1])

    weight_ups_FIN = agg.rule(torch.Tensor(weight_ups))  # aggregate weight grad
    bias_ups_FIN = agg.rule(torch.Tensor(bias_ups))  # aggregate bias grad

    central.update_model((weight_ups_FIN, bias_ups_FIN))

    # Evaluate model




# TODO: write model & optim & loss
