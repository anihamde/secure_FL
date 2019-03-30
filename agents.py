import torch
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
import torch.nn.functional as F

class Central():
	def __init__(self,model,optim):
		self.model = model
		self.optim = optim

	def update_model(self,ups): # ups is a tuple consisting of weight and bias grads
		self.model.weight.zero_grad()
		self.model.weight.grad = ups[0]
		self.model.bias.grad = ups[1]
		self.optim.step()
		self.model.weight.zero_grad()

class Worker():
	def __init__(self,loss):
		self.model = None
		self.loss = loss

	def fwd_bkwd(self,inp,outp):
		self.model.weight.zero_grad()

		pred = self.model(inp)
		lossval = self.loss(pred,outp)
		lossval.backward()

		return (self.model.weight.grad,self.model.bias.grad)

class Agg():
	def __init__(self,rule):
		self.rule = rule # rule should be a function that takes a list of gradient updates and aggregates them
