import inspect
import math
import os
import os.path as osp

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS


def module_classes(module):
    return [cls for class_name, cls in inspect.getmembers(module, inspect.isclass)]


def get_int_bits(inputs):
    if isinstance(inputs, torch.Tensor):
        inp_int_bits = [inputs.int_bits]
    elif isinstance(inputs, (list, tuple)):
        inp_int_bits = []
        for inp in inputs:
            inp_int_bits.extend(get_int_bits(inp))
    else:
        raise TypeError("Unexpected type of input: {}, "
                        "while tuple or torch.tensor required".format(inputs.__class__.__name__))
    return inp_int_bits


def set_int_bits(output, out_int_bits):
    if isinstance(output, torch.Tensor):
        output.int_bits = out_int_bits
    elif isinstance(output, tuple):
        for idx, _ in enumerate(output):
            output[idx].int_bits = out_int_bits
    else:
        raise TypeError("Unexpected type of output: {}, "
                        "while tuple or torch.tensor required".format(output.__class__.__name__))


def int_bits(inp, percent=0):
    if percent == 0:
        max_v = inp.abs().max().item()
    else:
        sorted_items, _ = torch.sort(inp.abs().view(-1), descending=True)
        max_v = sorted_items[int(sorted_items.shape[0] * percent)]
    return math.ceil(math.log(max_v, 2))


def integerize(inp, float_bits, bits=16):
    bound = math.pow(2.0, bits - 1)
    min_val = -bound
    max_val = bound - 1
    res = torch.floor_(inp * math.pow(2., float_bits) + 0.5)
    if (res > max_val).any() or (res < min_val).any():
        print('Overflow. Some values were clipped')
    return torch.clamp(res, min_val, max_val)


def roundnorm_reg(inp, num_rounded_bits):
    return torch.floor_((inp + math.pow(2., num_rounded_bits - 1)) * math.pow(2., -num_rounded_bits))


def gap8_clip(inp, bits):
    return torch.clamp(inp, -math.pow(2., bits), math.pow(2., bits) - 1)


def gap_round(inp, float_bits, bits=16):
    bound = math.pow(2.0, bits - 1)
    min_val = -bound
    max_val = bound - 1
    return torch.clamp(torch.floor_(inp * math.pow(2., float_bits) + 0.5), min_val, max_val) * math.pow(
        2., -float_bits)


class Folder(Dataset):
    def __init__(self, data_source, loader, transform):
        self.images_list = [
            osp.join(data_source, image_name)
            for image_name in os.listdir(data_source)
            if osp.splitext(image_name)[1].lower() in IMG_EXTENSIONS
        ]
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img = self.loader(self.images_list[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img
