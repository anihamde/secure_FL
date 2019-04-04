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

n_workers = 2
n_epochs = 10000
batch_size = 4
mean0_std = 0  # 0 if no zero-mean epsilon
learning_rate = 0.001

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Import Datasets
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

# Create Validation Split
valset = copy.deepcopy(trainset)
trainset.data = trainset.data[0:49000]
valset.data = valset.data[49000:50000]

# Create Train, Validation, and Test Loaders
sampler = torch.utils.data.RandomSampler(trainset, replacement=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False, sampler=sampler,
    num_workers=0)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=valset.data.shape[0], shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def rule(ups_list):  # ups_list is a list of list of tensors
    return [torch.stack([x[i] for x in ups_list]).mean(0)
            for i in range(len(ups_list[0]))]


def print_test_accuracy(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
          100 * correct / total))


# Setup Learning Model
model = SecondNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

# Setup Federated Learning Framework
central = Central(model, optimizer)
worker_list = []
for i in range(n_workers):
    worker_list.append(Worker(loss))
agg = Agg(rule)

e_dist_w = []
for layer, paramval in central.model.named_parameters():
    e_dist_w.append(Normal(torch.zeros_like(paramval), mean0_std))

# Training Loop
for t in range(n_epochs):
    weight_ups = []
    central.model.train()
    dataiter = iter(trainloader)

    # Worker Loop
    for i in range(n_workers):
        batch_inp, batch_outp = dataiter.next()

        worker_list[i].model = central.model

        ups = worker_list[i].fwd_bkwd(batch_inp, batch_outp)
        for i in range(len(e_dist_w)):
            ups[i] += e_dist_w[i].sample()

        weight_ups.append(ups)

    # Aggregate Worker Gradients
    weight_ups_FIN = agg.rule(weight_ups)

    # Update Central Model
    central.update_model(weight_ups_FIN)

    central.model.eval()

    if t % 500 == 0:
        print('Epoch: {}'.format(t))
        print_test_accuracy(model)
