import json
import logging
import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

from gap_quantization.layer_quantizers import LAYER_QUANTIZERS
from gap_quantization.utils import Folder, get_int_bits, int_bits, set_int_bits


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

    if not hasattr(module, 'out_int_bits') or out_int_bits > module.out_int_bits:
        module.out_int_bits = out_int_bits
    # propagate info through the network
    set_int_bits(output, out_int_bits)


class ModelQuantizer():
    def __init__(self, model, cfg, transform=None, layer_quantizers=None, loader=default_loader):
        self.model = model
        self.cfg = cfg
        if layer_quantizers is not None:
            self.layer_quantizers = layer_quantizers
        else:
            self.layer_quantizers = LAYER_QUANTIZERS
        self.transform = transform
        self.loader = loader

    def quantize_model(self):
        self.collect_stats()

        for name, module in self.model.named_modules():
            params = self.quantize_layer(module)
            if params is not None:
                for param_name in params:
                    value_to_set = torch.Tensor(params[param_name])
                    if self.cfg['use_gpu']:
                        value_to_set = value_to_set.cuda()
                    setattr(module, param_name, torch.nn.Parameter(value_to_set))
                if self.cfg['save_params']:
                    self.save_quant_params(params, name)

    def quantize_layer(self, module):
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
