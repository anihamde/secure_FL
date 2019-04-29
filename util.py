import numpy as np
import matplotlib.pyplot as plt
import torch
import os


def save_data(x, y, savefile):
    print(np.stack([x, y]))
    np.save(savefile, np.stack([x, y]))


def plot_data(x, y, xlabel=None, ylabel=None, title=None, savefile=None):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()

    if savefile is not None:
        plt.savefig(savefile)
    # plt.show()


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def print_test_accuracy(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
          100 * correct / total))
    return 100 * correct / total


def check_mem():

    mem = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split(",")

    return mem
