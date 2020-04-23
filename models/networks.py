import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class Identity(nn.Module):
    def forward(self, x):
        return x