import torch
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import torch.nn.functional as F


class Central():
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim

    def update_model(self, ups):
        """
        Update the central model with the new gradients.
        ups is consisting of weight grads
        """

        self.optim.zero_grad()
        i = 0
        for layer, paramval in self.model.named_parameters():
            paramval.grad = ups[i]
            i += 1
        self.optim.step()
        self.optim.zero_grad()


class Worker():
    def __init__(self, loss):
        self.model = None
        self.loss = loss

    def fwd_bkwd(self, inp, outp):
        pred = self.model(inp)
        lossval = self.loss(pred, outp)
        lossval.backward()

        weightgrads = []
        for layer, paramval in self.model.named_parameters():
            weightgrads.append(paramval.grad)
        return weightgrads


class Agg():
    def __init__(self, rule):
        self.rule = rule  # rule should be a function that takes a list of gradient updates and aggregates them
