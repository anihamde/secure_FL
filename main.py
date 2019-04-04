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
import copy

n_workers = 1
n_epochs = 100
batch_size = 16
mean0_std = 0  # 0 if no zero-mean epsilon
learning_rate = 0.01

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
valset = copy.deepcopy(trainset)
trainset.data = trainset.data[0:49000]
valset.data = valset.data[49000:50000]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=valset.data.shape[0], shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Setup Learning Model
model = SecondNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()


def rule(ups_list):  # ups_list is a list of list of tensors
    return [torch.stack([x[i] for x in ups_list]).mean(0)
            for i in range(len(ups_list[0]))]


# Setup Federated Learning Framework
central = Central(model, optimizer)
worker_list = []
for i in range(n_workers):
    worker_list.append(Worker(loss))
agg = Agg(rule)

e_dist_w = []
for layer, paramval in central.model.named_parameters():
    e_dist_w.append(Normal(torch.zeros_like(paramval), mean0_std))

for t in range(n_epochs):
    print('Epoch {}'.format(t))
    weight_ups = []
    central.model.train()
    for i in range(n_workers):
        dataiter = iter(trainloader)
        batch_inp, batch_outp = dataiter.next()

        worker_list[i].model = central.model

        ups = worker_list[i].fwd_bkwd(batch_inp, batch_outp)
        for i in range(len(e_dist_w)):
            ups[i] += e_dist_w[i].sample()

        weight_ups.append(ups)

    weight_ups_FIN = agg.rule(weight_ups)  # aggregate weight grad

    central.update_model(weight_ups_FIN)

    total = 0
    correct = 0
    central.model.eval()

    for data in valloader:
        inp, outp = data

        preds = central.model(inp)
        _, predicted_labs = torch.max(preds, 1)
        total += outp.size(0)
        correct += (predicted_labs == outp).sum().item()

    print('Accuracy of the network on val set: {}'.format(100 * correct / float(total)))
