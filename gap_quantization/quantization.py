import json
import logging
import math
import os
import os.path as osp
from functools import partial

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

import gap_quantization.quantized_layers
from gap_quantization.layer_quantizers import LAYER_QUANTIZERS
from gap_quantization.layers import Concat
from gap_quantization.quantized_layers import QUANTIZED_LAYERS
from gap_quantization.utils import (
    Folder,
    get_int_bits,
    int_bits,
    merge_batch_norms,
    module_classes,
    roundnorm_reg,
    set_param,
)


def save_act_hook(module, inp, output, save_dir, name, rounded):

    if isinstance(inp, (tuple, list)) and len(inp) == 1:
        inp = inp[0]
    try:
        inp = inp.data.cpu().numpy().tolist()
    except AttributeError:
        # if verbose:
        #     print("module {} was ignored".format(name))
        return

    if rounded:
        if hasattr(module, 'out_frac_bits'):
            output /= math.pow(2, module.out_frac_bits)
        else:
            return
    output = output.data.cpu().numpy().tolist()

    os.makedirs(os.path.join(save_dir, name), exist_ok=True)
    with open(os.path.join(save_dir, name, 'input.json'), 'w') as inp_f, \
            open(os.path.join(save_dir, name, 'output.json'), 'w') as out_f:
        json.dump(inp, inp_f)
        json.dump(output, out_f)


def stats_hook(module, inputs, output):
    inp_int_bits = get_int_bits(inputs)

    if not hasattr(module, 'inp_int_bits'):
        module.inp_int_bits = inp_int_bits
    else:
        for idx, (curr_inp_int_bits, new_inp_int_bits) in enumerate(zip(module.inp_int_bits, inp_int_bits)):
            if new_inp_int_bits > curr_inp_int_bits:
                module.inp_int_bits[idx] = new_inp_int_bits

    if isinstance(module, nn.Conv2d):
        out_int_bits = int_bits(output)
    else:
        out_int_bits = max(inp_int_bits)

    if module.__class__ in module_classes(nn) \
            and not isinstance(module, (nn.Sequential, nn.ModuleList)) \
            or module.__class__ in module_classes(gap_quantization.layers):
        # ignore custom modules: Fire, Bottleneck, ..., high-level PyTorch modules
        if not hasattr(module, 'out_int_bits') or out_int_bits > module.out_int_bits:
            module.out_int_bits = out_int_bits
        # propagate info through the network
        set_param(output, 'int_bits', out_int_bits)


def first_element(tensor):
    return tensor.view(-1)[0].item()


def getattr_rec(obj, attr_list):
    if attr_list:  # empty sequence is False in Python
        return getattr_rec(getattr(obj, attr_list[0]), attr_list[1:])
    return obj


def shift_concat_input(module, grad_input, grad_output):
    if isinstance(module, nn.Conv2d) and first_element(grad_output[0]):
        shift = first_element(grad_output[0])
        print('shifted module {} by {} bits'.format(module.__class__.__name__, shift))
        module.norm += shift
        module.bias = nn.Parameter(roundnorm_reg(module.bias, shift))
        grad_input = tuple(torch.zeros_like(tensor) for tensor in grad_input)
    elif grad_output[0].sum() != 0:
        tmp = []
        for tensor in grad_input:
            if tensor is not None:
                tmp.append(torch.empty_like(tensor).fill_(first_element(grad_output[0])))
            else:
                tmp.append(None)
        grad_input = tuple(tmp)
        print('propagated through {}'.format(module.__class__.__name__))
    elif isinstance(module, Concat):
        print(module.norm)
        tmp = []
        for curr_norm, tensor in zip(module.norm, grad_input):
            tmp.append(torch.empty_like(tensor).fill_(curr_norm))
        module.norm = nn.Parameter(torch.Tensor([0 for _ in module.norm]))
        grad_input = tuple(tmp)
    elif module.__class__ in module_classes(nn) \
            and not isinstance(module, (nn.Sequential, nn.ModuleList)) \
            or module.__class__ in module_classes(gap_quantization.layers):
        tmp = []
        for tensor in grad_input:
            if tensor is not None:
                tmp.append(torch.zeros_like(tensor))
            else:
                tmp.append(None)  # for convolutions without biases
    return grad_input


class ModelQuantizer():
    def __init__(self,
                 model,
                 cfg,
                 transform=None,
                 layer_quantizers=None,
                 quantized_layers=None,
                 loader=default_loader):
        self.model = model
        self.cfg = cfg
        if layer_quantizers is not None:
            self.layer_quantizers = layer_quantizers
        else:
            self.layer_quantizers = LAYER_QUANTIZERS
        if quantized_layers is not None:
            self.quantized_layers = quantized_layers
        else:
            self.quantized_layers = QUANTIZED_LAYERS
        self.transform = transform
        self.loader = loader
        self.params_dict = {}

    def quantize_model(self):
        merge_batch_norms(self.model)
        self.collect_stats()
        self.quantize_parameters_rec(self.model)
        self.set_parameters_rec(self.model)
        self.fix_alignment()

        if self.cfg['quantize_forward']:
            self.quantize_forward_rec(self.model)
            self.set_parameters_rec(self.model)

        if self.cfg['save_params']:
            for name in self.params_dict:
                self.save_quant_params(self.params_dict[name], name)

    def fix_alignment(self):
        # fix cpu-gpu conversion
        backward_hooks = []
        for module in self.model.modules():
            backward_hooks.append(module.register_backward_hook(shift_concat_input))

        inp = torch.rand((1, self.cfg['num_input_channels'], 224, 224))
        out = self.model(inp)
        self.model.zero_grad()
        out.backward(torch.zeros_like(out))

        for handle in backward_hooks:  # delete forward hooks
            handle.remove()
        # update param_dict
        for full_module_name in self.params_dict:
            for param_name in self.params_dict[full_module_name]:
                new_value = getattr_rec(self.model, full_module_name.split('.') + [param_name])
                if not torch.equal(torch.Tensor(self.params_dict[full_module_name][param_name]), new_value):
                    if self.cfg['verbose']:
                        print('Parameter {} for module {} was fixed'.format(param_name, full_module_name))
                    self.params_dict[full_module_name][param_name] = new_value.data.cpu().numpy().tolist()

    def quantize_parameters_rec(self, module, module_name=None):
        for name, submodule in module.named_children():
            params = self.quantize_parameters(submodule)
            if module_name is not None:
                full_module_name = '.'.join([module_name, name])
            else:
                full_module_name = name
            if params is not None:
                self.params_dict[full_module_name] = params

                if self.cfg['verbose']:
                    print(full_module_name)
                    out = ''
                    for k, val in params.items():
                        if 'norm' in k or 'bits' in k and not isinstance(submodule, Concat):
                            out += '{}: {}, '.format(k, val)
                    out += '\n'
                    print(out)
            self.quantize_parameters_rec(submodule, full_module_name)

    def quantize_forward_rec(self, module, module_name=None):
        for name, submodule in module.named_children():
            if module_name is not None:
                full_module_name = '.'.join([module_name, name])
            else:
                full_module_name = name
            submodule = self.quantize_forward(submodule)
            # if '.'.join([module_name, name]) in self.params_dict:
            #     for param_name in self.params_dict['.'.join([module_name, name])]:
            #         value_to_set = torch.Tensor(self.params_dict['.'.join([module_name, name])][param_name])
            #         setattr(submodule, param_name, torch.nn.Parameter(value_to_set))
            try:
                setattr(module, name, submodule)
            except AttributeError:
                if self.cfg['verbose']:
                    print('Attribute {} wasn\'t set for {}'.format(name, module_name))
            self.quantize_forward_rec(submodule, full_module_name)

    def set_parameters_rec(self, module, module_name=None):
        for name, submodule in module.named_children():
            if module_name is not None:
                full_module_name = '.'.join([module_name, name])
            else:
                full_module_name = name
            if full_module_name in self.params_dict:
                for param_name in self.params_dict[full_module_name]:
                    value_to_set = torch.Tensor(self.params_dict[full_module_name][param_name])
                    setattr(submodule, param_name, torch.nn.Parameter(value_to_set))
            try:
                setattr(module, name, submodule)
            except AttributeError:
                if self.cfg['verbose']:
                    print('Attribute {} wasn\'t set for {}'.format(name, module_name))
            self.set_parameters_rec(submodule, full_module_name)

    def quantize_forward(self, module):
        if module.__class__ in self.quantized_layers:
            args_dict = {arg: module.__dict__[arg] for arg in module.__dict__ if arg[0] != '_'}
            args_dict['bits'] = self.cfg['bits']
            quantized_layer = self.quantized_layers[module.__class__](**args_dict)
            for arg in args_dict:
                setattr(quantized_layer, arg, args_dict[arg])
            return quantized_layer
        return module

    def quantize_parameters(self, module):
        if module.__class__ in self.layer_quantizers:
            return self.layer_quantizers[module.__class__](module, self.cfg)
        return None

    def save_quant_params(self, params, name):
        os.makedirs(self.cfg['save_folder'], exist_ok=True)

        with open(osp.join(self.cfg['save_folder'], name + '.json'), 'w') as f:
            json.dump(params, f)

    def collect_stats(self):
        handles = []
        for module in self.model.modules():
            handles.append(module.register_forward_hook(stats_hook))

        dataset = Folder(self.cfg['data_source'], self.loader, self.transform)

        if self.cfg['verbose']:
            logging.info('{} images are used to collect statistics'.format(len(dataset)))

        dataloader = DataLoader(dataset,
                                batch_size=self.cfg['batch_size'],
                                shuffle=False,
                                num_workers=self.cfg['num_workers'],
                                drop_last=False)
        self.model.eval()

        if self.cfg['use_gpu']:
            self.model.cuda()

        with torch.no_grad():
            for imgs in tqdm(dataloader):
                if self.cfg['raw_input']:
                    imgs.int_bits = self.cfg['bits'] - self.cfg['signed']
                else:
                    imgs.int_bits = int_bits(imgs)
                if self.cfg['use_gpu']:
                    imgs = imgs.cuda()
                _ = self.model(imgs)

        for handle in handles:  # delete forward hooks
            handle.remove()

        if self.cfg['use_gpu']:
            self.model.cpu()

    def dump_activations(self, image_path, transforms, rounded=False, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(self.cfg['save_folder'], 'activation_dump')
        handles = []
        for name, module in self.model.named_modules():
            if module.__class__ not in module_classes(gap_quantization.layers):
                hook = partial(save_act_hook, save_dir=save_dir, name=name, rounded=rounded)
                handles.append(module.register_forward_hook(hook))

        self.model.eval()

        img = Image.open(image_path).convert('RGB')
        tensor = transforms(img)

        if self.cfg['use_gpu']:
            tensor = tensor.cuda()

        _ = self.model(tensor.unsqueeze_(0)).data.cpu().tolist()

        for handle in handles:  # delete forward hooks
            handle.remove()
