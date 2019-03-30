import torch
from torch.autograd import Variable
from torch.distributions.normal import Normal
import torch.nn.functional as F
import numpy as np
from agents import *
from models import *

n_workers = 80
n_epochs = 100
mean0_std = 0 # 0 if no zero-mean epsilon



model = 
optim = 
loss = 
def rule(ups_list): # ups_list is a tensor
	return mean(ups_list)[0]

central = Central(model,optim)
worker_list = []
for i in range(n_workers):
	worker_list.append(Worker(loss))
agg = Agg(rule)

e_dist_w = Normal(torch.zeros_like(central.model.weight),mean0_std)
e_dist_b = Normal(torch.zeros_like(central.model.bias),mean0_std)



for t in range(n_epochs):
	weight_ups = []
	bias_ups = []
	for i in range(n_workers):
		batch =  # get random batch subset of dataset
		batch_inp = 
		batch_outp = 
		worker_list[i].model = central.model
		ups = worker_list[i].fwd_bkwd(batch_inp,batch_outp)
		ups[0] += e_dist_w.sample()
		ups[1] += e_dist_b.sample()
		weight_ups.append(ups[0])
		bias_ups.append(ups[1])

	weight_ups_FIN = agg.rule(torch.Tensor(weight_ups)) # aggregate weight grad
	bias_ups_FIN = agg.rule(torch.Tensor(bias_ups)) # aggregate bias grad

	central.update_model((weight_ups_FIN,bias_ups_FIN))

	## Evaluate model




# TODO: get datasets configured here, write random batch selection code, write model & optim & loss

