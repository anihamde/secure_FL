"""
Securing Federated Learning: Obfuscation and Encryption

Lev Grossman and Anirudh Suresh
lgrossman@college.harvard.edu, anirudh_suresh@college.harvard.edu


TODO: iid vs. non-iid data during training (split into class-specific)
"""

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
from util import *
import copy
import time
import sys

n_workers = 10
n_epochs = 1
batch_size = 64
mean0_std = 0  # 0 if no zero-mean epsilon
learning_rate = 0.001
encrypt = False
save_data_and_plots = False
noniid = True

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
advset = copy.deepcopy(trainset)
trainset.data = trainset.data[0:48000]
trainset.targets = trainset.targets[0:48000]

valset.data = valset.data[48000:49000]
valset.targets = valset.targets[48000:49000]

advset.data = advset.data[49000:50000]
advset.targets = advset.targets[49000:50000]

# Create Train, Validation, and Test Loaders
if noniid:
	def noniid_batch_trainset(trainset, c):
		indices = (np.array(trainset.targets)==c)
		trainset2 = copy.deepcopy(trainset)
		trainset2.data = trainset2.data[indices]
		trainset2.targets = [c for i in range(len(indices))]

		return trainset2

	trainsets = [noniid_batch_trainset(trainset,i) for i in set(trainset.targets)]
else:
	trainsets = [trainset]

samplers = [torch.utils.data.RandomSampler(i, replacement=True) for i in trainsets]
trainloaders = [torch.utils.data.DataLoader(
    trainsets[i], batch_size=batch_size, shuffle=False, sampler=samplers[i],
    num_workers=0) for i in range(len(trainsets))]

valloader = torch.utils.data.DataLoader(
    valset, batch_size=valset.data.shape[0], shuffle=False, num_workers=0)

# advloader = torch.utils.data.DataLoader(
#     advset, batch_size=batch_size, shuffle=False, num_workers=0)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0)


def encrypted_rule(ups_list):
    return [np.stack([x[i] for x in ups_list]).mean(0)
            for i in range(len(ups_list[0]))]


def rule(ups_list):  # ups_list is a list of list of tensors
    return [torch.stack([x[i] for x in ups_list]).mean(0)
            for i in range(len(ups_list[0]))]


# Setup Learning Model
model = PerformantNet1()
# model = torchvision.models.vgg16_bn()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")
print(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()

# Setup Federated Learning Framework
central = Central(model, optimizer, encryption=encrypt)
worker_list = []
for i in range(n_workers):
    worker_list.append(Worker(loss, key=central.get_key()))
if encrypt:
    agg = Agg(encrypted_rule)
else:
    agg = Agg(rule)

e_dist_w = []
for layer, paramval in central.model.named_parameters():
    e_dist_w.append(Normal(torch.zeros_like(paramval), mean0_std))

epochs = []
accuracies = []

# Training Loop
for t in range(n_epochs):
    first_time = time.time()

    weight_ups = []
    central.model.train()

    dataiters = [iter(trainloader) for trainloader in trainloaders]

    # Worker Loop
    for i in range(n_workers):
        k = np.random.randint(0,len(dataiters))
        dataiter = dataiters[k]

        batch_inp, batch_outp = dataiter.next()
        batch_inp, batch_outp = batch_inp.to(device), batch_outp.to(device)

        worker_list[i].model = central.model

        ups = worker_list[i].fwd_bkwd(batch_inp, batch_outp)

        if not encrypt:
            for i in range(len(e_dist_w)):
                ups[i] += e_dist_w[i].sample()

        weight_ups.append(ups)

    # Aggregate Worker Gradients
    weight_ups_FIN = agg.rule(weight_ups)

    # Update Central Model
    central.update_model(weight_ups_FIN)

    central.model.eval()

    if t > 0 and t % 100 == 0:
        print('Epoch: {}, Time to complete: {}'.format(t, time.time() - first_time))

    if t % 250 == 0:
        # print('Epoch: {}'.format(t))
        accuracy = print_test_accuracy(model, testloader)
        epochs.append(t)
        accuracies.append(accuracy)

print('Done training')

exit()

total, used = check_mem()
total, used = int(total), int(used)
print(total), print(used)



if save_data_and_plots:
    savefile = "plots/workers={}_batch_size={}_std={}_lr={}_epochs={}.png".format(
        n_workers, batch_size, mean0_std, learning_rate, n_epochs)
    save_data(epochs, accuracies, savefile)
    plot_data(epochs, accuracies, xlabel="epoch", ylabel="accuracy",
              savefile=savefile)
    print('Done saving data')

# Adversarial attack


paramslist = list(central.model.parameters())
paramslist = [x.view(-1) for x in paramslist]
paramslist = torch.cat(paramslist)

learning_rate_adv = 0.01
n_epochs_adv = 20
adv_model = AdvNet(paramslist.shape[0])
adv_model.to(device)
central.init_adv(adv_model)
adv_optim = optim.Adam(central.adv.parameters(), lr=learning_rate_adv)

adv_dataset = []

for j in range(len(advset.data)):
    optimizer.zero_grad()

    x = torch.Tensor(advset.data[j]).transpose(-1, -2).transpose(-2, -3).unsqueeze(0).cuda()
    y = torch.LongTensor([advset.targets[j]]).cuda()
    # x, y = x.to(device), y.to(device)
    # x_cuda = x.to(device)
    x = central.model(x)
    # x = x.to(cpu_device)

    lossval = loss(x, y)

    lossval.backward()

    weightgrads = []
    for layer, paramval in central.model.named_parameters():
        weightgrads.append(paramval.grad.flatten())

    weightgrads = torch.cat(weightgrads)

    adv_dataset.append([weightgrads, y.to(cpu_device)])
    del x, y
    # del x_cuda
    # del x

    if j % 100 == 0:
        torch.cuda.empty_cache()
        total, used = check_mem()
        total, used = int(total), int(used)
        print(total), print(used)
        print('emptied')

optimizer.zero_grad()

torch.cuda.empty_cache()
print('HERE')

adv_x = torch.stack([x[0] for x in adv_dataset])
adv_y = torch.stack([x[1] for x in adv_dataset])

adv_dataset = torch.utils.data.TensorDataset(adv_x,adv_y)

advloader = torch.utils.data.DataLoader(
    adv_dataset, batch_size=batch_size, shuffle=True, #sampler=sampler,
    num_workers=0)

# Spliced this block
test_adv_dataset = []

for j in range(len(testset.data)):
    optimizer.zero_grad()

    x = torch.Tensor(testset.data[j]).transpose(-1, -2).transpose(-2, -3).unsqueeze(0)
    y = torch.LongTensor([testset.targets[j]])
    x = x.to(device)
    x = central.model(x)
    x = x.to(cpu_device)
    # x, y = x.to(device), y.to(device)

    lossval = loss(x, y)

    lossval.backward()

    weightgrads = []
    for layer, paramval in central.model.named_parameters():
        weightgrads.append(paramval.grad.flatten())

    weightgrads = torch.cat(weightgrads)

    test_adv_dataset.append([weightgrads + Normal(torch.zeros_like(weightgrads), mean0_std).sample(), y])

optimizer.zero_grad()

test_adv_x = torch.stack([x[0] for x in test_adv_dataset])
test_adv_y = torch.stack([x[1] for x in test_adv_dataset])

# test_adv_x, test_adv_y = test_adv_x.to(device), test_adv_y.to(device)
# End splice



for t in range(n_epochs_adv):
    central.adv.eval()

    pred_labels = torch.argmax(central.adv(test_adv_x),1)

    total_correct = (pred_labels == test_adv_y.squeeze()).sum()

    print('Sum: {}'.format(total_correct))
    print('Length: {}'.format(len(pred_labels)))

    print(float(total_correct)/float(len(pred_labels)))

    print('Adv Epoch: {}'.format(t))

    adv_optim.zero_grad()

    central.adv.train()

    for i_batch, sample_batched in enumerate(advloader):
        batch_inp, batch_outp = sample_batched
        # batch_inp, batch_outp = batch_inp.to(device), batch_outp.to(device)
        preds = central.adv(batch_inp)

        lossval = loss(preds, batch_outp.squeeze())
        lossval.backward()

    adv_optim.step()

    adv_optim.zero_grad()




central.adv.eval()

pred_labels = torch.argmax(central.adv(test_adv_x),1)

total_correct = (pred_labels == test_adv_y.squeeze()).sum()

print('Sum: {}'.format(total_correct))
print('Length: {}'.format(len(pred_labels)))

print(float(total_correct)/float(len(pred_labels)))




###