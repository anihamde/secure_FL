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
import time
# import SharedArray as sa


def test_training():
    n_workers = 1
    n_epochs = 100000
    batch_size = 4
    mean0_std = 0  # 0 if no zero-mean epsilon
    learning_rate = 0.001

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valset = copy.deepcopy(trainset)
    trainset.data = trainset.data[0:49000]
    valset.data = valset.data[49000:50000]


    sampler = torch.utils.data.RandomSampler(trainset, replacement=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)



    # Setup Learning Model
    model = SecondNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

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

    print('Finished Training')


def test_stacking():
    model = FirstNet()

    lst = []
    for layer, params in model.named_parameters():
        print(layer)
        print(params.shape)
        print(params.grad)
        params.grad = torch.zeros_like(params)
        lst.append(params.grad)
        print(params.grad)

    new_lst = [lst, lst]
    print(new_lst)
    tmp = torch.stack([x[0] for x in new_lst])
    print(tmp)
    print(tmp.mean(0).shape)
    print(tmp.shape)


def test_phe():
    from phe import paillier
    print('Testing phe...')

    keyring = paillier.PaillierPrivateKeyring()
    public_key, private_key = paillier.generate_paillier_keypair(keyring)

    nums = np.array([2., 3.1, 18])
    encrypted_nums = [public_key.encrypt(x) + 1 for x in nums]
    decrypted_nums = [keyring.decrypt(x) for x in encrypted_nums]

    print(nums)
    print(encrypted_nums)
    print(np.array(encrypted_nums).dtype)
    print(decrypted_nums)

    mat = np.array([[1., 2.], [3., 4.]])
    mat_e = np.zeros(mat.shape, dtype=object)

    print(mat)
    print(mat_e)

    for index, x in np.ndenumerate(mat):
        mat_e[index] = public_key.encrypt(float(x))

    print(mat_e)


class Test(object):
    """docstring for ClassName"""
    def __init__(self, arg):
        super(Test, self).__init__()
        self.arg = arg


class Gradient(object):
    """docstring for ClassName"""
    def __init__(self):
        super(Gradient, self).__init__()
        self.grads = np.random.rand(400, 120)
        self.grads_e = np.zeros(self.grads.shape, dtype=object)

    def reset_grads_e(self):
        self.grads_e = np.zeros(self.grads.shape, dtype=object)


def encrypt(args):
    key, index, x, grads_e = args
    # grads_e[index] = key.encrypt(float(x))
    return key.encrypt(float(x))

    # b = sa.attach("shm://test")
    # b[index] = Test(float(x))


def test_mp():
    from multiprocessing import Array, Manager, RawArray
    import multiprocessing as mp
    from phe import paillier

    keyring = paillier.PaillierPrivateKeyring()
    public_key, private_key = paillier.generate_paillier_keypair(keyring, n_length=128)

    g = Gradient()
    grads = g.grads
    print(grads.shape)

    begin_time = time.time()
    grads_e = np.zeros(grads.shape, dtype=object)
    for index, x in np.ndenumerate(grads):
        grads_e[index] = public_key.encrypt(float(x))
    end_time = time.time()

    print('Base time: {}'.format(end_time - begin_time))

    # Multiprocessing
    grads_e = np.zeros(grads.shape, dtype=object)

    pool = mp.Pool(mp.cpu_count())

    grads_e = np.zeros(grads.shape, dtype=object)

    # sa.delete("shm://test")
    # a = sa.create("shm://test", grads.shape, dtype=object)
    # print(a)

    begin_time = time.time()
    nargs = [(public_key, index, x, grads_e) for index, x in np.ndenumerate(grads)]
    grads_e = np.reshape(pool.map(encrypt, nargs), grads.shape)
    end_time = time.time()

    print('MP time: {}'.format(end_time - begin_time))
    pool.close()

    # print(a)
    # sa.delete("shm://test")


if __name__ == '__main__':
    test_mp()
