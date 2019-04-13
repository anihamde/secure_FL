import numpy as np
import matplotlib.pyplot as plt


def plot_data(x, y, xlabel=None, ylabel=None, title=None, savefile="tmp.png"):
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid()

    plt.savefig(savefile)
    # plt.show()
