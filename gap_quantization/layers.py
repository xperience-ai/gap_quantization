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


class View(nn.Module):
    def forward(self, inp, shapes):
        return inp.view(*shapes)


class Flatten(nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, inputs):
        return torch.flatten(inputs, self.start_dim, self.end_dim)
