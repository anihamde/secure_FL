import torch
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import torch.nn.functional as F
import numpy as np
from phe import paillier


class Central():
    def __init__(self, model, optim, encryption=False):
        self.model = model
        self.optim = optim

        # Encryption-based Setup
        self.keyring = None
        if encryption:
            self.keyring = paillier.PaillierPrivateKeyring()

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

    def init_adv(self, model):
        self.adv = model

    def get_keyring(self):
        """
        Returns an encryption keyring to store public/private keys
        """
        return self.keyring


class Worker():
    def __init__(self, loss, keyring=None):
        self.model = None
        self.loss = loss
        self.keyring = keyring
        if keyring is not None:
            self.public_key, self.private_key = (
                paillier.generate_paillier_keypair(keyring, n_length=128))

    def fwd_bkwd(self, inp, outp):
        pred = self.model(inp)
        lossval = self.loss(pred, outp)
        lossval.backward()

        weightgrads = []
        for layer, paramval in self.model.named_parameters():
            if self.keyring is not None:
                grads = np.array(paramval.grad)
                print(grads.shape)

                grads_e = np.zeros(grads.shape, dtype=object)
                for index, x in np.ndenumerate(grads):
                    grads_e[index] = self.public_key.encrypt(float(x))

                weightgrads.append(paramval.grad)
            else:
                weightgrads.append(paramval.grad)

        return weightgrads


class Agg():
    def __init__(self, rule):
        self.rule = rule  # rule should be a function that takes a list of gradient updates and aggregates them
