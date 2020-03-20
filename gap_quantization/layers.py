import torch
import torch.nn as nn


class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, self.dim)


class Stack(nn.Module):
    def __init__(self, dim):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.stack(inputs, self.dim)


class EltWiseMul(nn.Module):
    def forward(self, inp1, inp2):
        return inp1 * inp2


class EltWiseAdd(nn.Module):
    def __init__(self):
        super(EltWiseAdd, self).__init__()

    def forward(self, inputs):
        res = inputs[0]
        for tensor in inputs[1:]:
            res += tensor
        return res


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


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, inp):
        return inp.squeeze(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, inp):
        return inp.unsqueeze(self.dim)


class Chunk(nn.Module):
    def __init__(self, num_chunks, dim):
        super(Chunk, self).__init__()
        self.num_chunks = num_chunks
        self.dim = dim

    def forward(self, inp):
        return torch.chunk(inp, self.num_chunks, self.dim)


class SplitSqueeze(nn.Module):
    def __init__(self, num_chunks, dim):
        super(SplitSqueeze, self).__init__()
        self.num_chunks = num_chunks
        self.dim = dim

    def forward(self, inp):
        return [tensor.squeeze(self.dim) for tensor in torch.split(inp, self.num_chunks, self.dim)]
