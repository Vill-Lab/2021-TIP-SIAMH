# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on:
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py


import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fastreid.utils import comm

from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.solver.optim import Adam


from fastreid.solver import lr_scheduler
from fastreid.solver import optim


def build_optimizer_gan(model,OPT="Adam",MOMENTUM=0.9,BASE_LR=0.00035,WEIGHT_DECAY=0.0005,BIAS_LR_FACTOR=2.0,WEIGHT_DECAY_BIAS=0.0005):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = BASE_LR
        weight_decay = WEIGHT_DECAY
        if "heads" in key:
            lr *= HEADS_LR_FACTOR
        if "bias" in key:
            lr *= BIAS_LR_FACTOR
            weight_decay =WEIGHT_DECAY_BIAS
        params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay, "freeze": False}]

    solver_opt = OPT
    if hasattr(optim, solver_opt):
        if solver_opt == "SGD":
            opt_fns = getattr(optim, solver_opt)(params, momentum=MOMENTUM)
        else:
            opt_fns = getattr(optim, solver_opt)(params)
    else:
        raise NameError("optimizer {} not support".format(OPT))
    return opt_fns
from torch import nn

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class discritor_code(nn.Module):
    def __init__(self,infeat_dim):
        super(discritor_code,self).__init__()
        self.fc1 = nn.Linear(infeat_dim,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,2)
        self.swish = MemoryEfficientSwish()
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.swish(x)
        x = self.fc2(x)
        x = self.swish(x)
        x = self.fc3(x)
        x = self.swish(x)
        x = self.fc4(x)
        return x  
# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)






class discritor_cam(nn.Module):
    def __init__(self,infeat_dim,camidnum):
        super(discritor_cam,self).__init__()
        self.fc1 = nn.Linear(infeat_dim,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,camidnum)
        self.swish = MemoryEfficientSwish()
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.swish(x)
        x = self.fc2(x)
        x = self.swish(x)
        x = self.fc3(x)
        x = self.swish(x)
        x = self.fc4(x)
        return x  