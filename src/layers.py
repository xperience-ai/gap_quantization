import torch
import torch.nn as nn


class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, self.dim)


class EltWiseMul(nn.Module):
    def forward(self, inp1, inp2):
        return inp1 * inp2


class EltWiseAdd(nn.Module):
    def forward(self, inp1, inp2):
        return inp1 + inp2