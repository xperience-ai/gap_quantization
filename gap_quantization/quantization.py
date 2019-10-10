import json
import logging
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

import gap_quantization.quantized_layers
from gap_quantization.layer_quantizers import LAYER_QUANTIZERS
from gap_quantization.quantized_layers import QUANTIZED_LAYERS
from gap_quantization.utils import Folder, get_int_bits, int_bits, module_classes, set_int_bits


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
        set_int_bits(output, out_int_bits)


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

    def quantize_model(self):
        self.collect_stats()
        self.quantize_module(self.model, 'net')

    def quantize_module(self, module, module_name):
        for name, submodule in module.named_children():
            params = self.quantize_parameters(submodule)
            if self.cfg['quantize_forward']:
                submodule = self.quantize_forward(submodule)
            if params is not None:
                for param_name in params:
                    value_to_set = torch.Tensor(params[param_name])
                    if self.cfg['use_gpu']:
                        value_to_set = value_to_set.cuda()
                    setattr(submodule, param_name, torch.nn.Parameter(value_to_set))
                if self.cfg['save_params']:
                    self.save_quant_params(params, name)
            try:
                setattr(module, name, submodule)
            except AttributeError:
                if self.cfg['verbose']:
                    print('Attribute {} wasn\'t set for {}'.format(name, module_name))
        for child_name, child in module.named_children():
            self.quantize_module(child, child_name)

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
                imgs.int_bits = int_bits(imgs)
                if self.cfg['use_gpu']:
                    imgs = imgs.cuda()
                _ = self.model(imgs)

        if self.cfg['use_gpu']:
            self.model.cpu()

        for handle in handles:  # delete forward hooks
            handle.remove()
