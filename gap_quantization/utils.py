import inspect
import math
import os
import os.path as osp
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
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


def set_param(tensor, name, val):
    if isinstance(tensor, torch.Tensor):
        setattr(tensor, name, val)
    elif isinstance(tensor, tuple):
        for idx, _ in enumerate(tensor):
            setattr(tensor[idx], name, val)
    else:
        raise TypeError("Unexpected type of output: {}, "
                        "while tuple or torch.tensor required".format(tensor.__class__.__name__))


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


def absorb_bn(module, bn_module):
    w = module.weight.data
    if module.bias is None:
        zeros = torch.Tensor(module.out_channels).zero_().type(w.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    invstd = bn_module.running_var.clone().add_(bn_module.eps).pow_(-0.5)
    w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
    bias.add_(-bn_module.running_mean).mul_(invstd)

    if bn_module.affine:
        w.mul_(bn_module.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
        bias.mul_(bn_module.weight.data).add_(bn_module.bias.data)

    bn_module.register_buffer('running_mean',
                              torch.zeros(bn_module.running_mean.data.size(), dtype=torch.float32))
    bn_module.register_buffer('running_var', torch.ones(bn_module.running_var.data.size(),
                                                        dtype=torch.float32))
    bn_module.register_parameter('weight', None)
    bn_module.register_parameter('bias', None)
    bn_module.affine = False


def is_bn(module):
    return isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))


def is_absorbing(module):
    return isinstance(module, nn.Conv2d)


def merge_batch_norms(model):
    prev = None
    for module in model.children():
        if is_bn(module) and is_absorbing(prev):
            absorb_bn(prev, module)
        merge_batch_norms(module)
        prev = module


def load_weights(network, save_path, partial=True):
    if urlparse(save_path).scheme != '':
        pretrained_dict = model_zoo.load_url(save_path)
    else:
        pretrained_dict = torch.load(save_path)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    try:
        network.load_state_dict(pretrained_dict)
        print('Pretrained network has absolutely the same layers!')
    except:
        model_dict = network.state_dict()
        pretrained_dict = {key: val for key, val in pretrained_dict.items() if key in model_dict}
        try:
            network.load_state_dict(pretrained_dict)
            print('Pretrained network has excessive layers; Only loading layers that are used')
        except:
            print('Pretrained network has fewer layers; The following are not initialized:')
            not_initialized = set()
            partially_initialized = set()
            for key, val in pretrained_dict.items():
                if val.size() == model_dict[key].size():
                    model_dict[key] = val
                elif partial:
                    min_shape = [
                        min(val.size()[i], model_dict[key].size()[i])
                        for i in range(len(min(val.size(), model_dict[key].size())))
                    ]
                    if len(model_dict[key].size()) in [2, 4]:  # fc and conv layers
                        model_dict[key][:min_shape[0], :min_shape[1], ...] = \
                            val[:min_shape[0], :min_shape[1], ...]
                    elif len(model_dict[key].size()) == 1:
                        model_dict[key][:min_shape[0]] = val[:min_shape[0]]
                    else:
                        print('{} has size: {}'.format(key, 'x'.join(model_dict[key].size())))

            for key, val in model_dict.items():
                if key not in pretrained_dict or (not partial and val.size() != pretrained_dict[key].size()):
                    not_initialized.add(key[:key.rfind('.')])
                elif partial and val.size() != pretrained_dict[key].size():
                    partially_initialized.add(key[:key.rfind('.')])
            print(sorted(not_initialized))
            if partial:
                print('Partially initialized:')
                print(sorted(partially_initialized))
            network.load_state_dict(model_dict)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
