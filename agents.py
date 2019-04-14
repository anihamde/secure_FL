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
            self.public_key, self.private_key = (
                paillier.generate_paillier_keypair(self.keyring, n_length=128))

    def update_model(self, ups):
        """
        Update the central model with the new gradients.
        ups is consisting of weight grads
        """

        self.optim.zero_grad()
        i = 0
        for layer, paramval in self.model.named_parameters():
            if self.keyring:
                print('Decrypting {} ...'.format(ups[i].shape))
                update = np.zeros(ups[i].shape)
                for index, x in np.ndenumerate(ups[i]):
                    try:
                        update[index] = self.private_key.decrypt(x)
                    except OverflowError as e:
                        print(x)

                update = torch.FloatTensor(update)
                paramval.grad = update
            else:
                paramval.grad = ups[i]
            i += 1
        self.optim.step()
        self.optim.zero_grad()

    def init_adv(self, model):
        self.adv = model

    def get_key(self):
        """
        Returns an encryption keyring to store public/private keys
        """
        if self.keyring:
            return self.public_key
        return None


class Worker():
    def __init__(self, loss, key=None):
        self.model = None
        self.loss = loss
        self.key = key

    def fwd_bkwd(self, inp, outp):
        pred = self.model(inp)
        lossval = self.loss(pred, outp)
        lossval.backward()

        weightgrads = []
        for layer, paramval in self.model.named_parameters():
            if self.key is not None:
                grads = np.array(paramval.grad)
                print('Encrypting {} ...'.format(grads.shape))

                grads_e = np.zeros(grads.shape, dtype=object)
                for index, x in np.ndenumerate(grads):
                    grads_e[index] = self.key.encrypt(float(x))

                weightgrads.append(grads_e)
            else:
                weightgrads.append(paramval.grad)

        return weightgrads


class Agg():
    def __init__(self, rule):
        self.rule = rule  # rule should be a function that takes a list of gradient updates and aggregates them
