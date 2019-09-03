from functools import partial
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets.folder import default_loader

from src.utils import int_bits, Folder
from src.layer_quantizers import layer_quantizers


def stats_hook(module, input, output):
    module.out_int_bits = int_bits(output)

    if isinstance(input, torch.Tensor):
        module.inp_int_bits = [int_bits(input)]
    if isinstance(input, tuple):
        inp_int_bits = []
        for inp in input:
            inp_int_bits.append(int_bits(inp))
        module.inp_int_bits = inp_int_bits


class ModelQuantizer():
    def __init__(self, model, cfg, transform=None, layer_quantizers=layer_quantizers, loader=default_loader):
        self.model = model
        self.cfg = cfg
        self.layer_quantizers = layer_quantizers
        self.transform = transform
        self.loader = loader

    def quantize_model(self):
        self.collect_stats()

        for name, module in self.model.named_modules():
            params = self.quantize_layer(module)
            if self.cfg['verbose']:
                raise NotImplementedError
            if params is not None:
                self.save_quant_params(params, name)

    def quantize_layer(self, module):
        if module.__class__ in self.layer_quantizers:
            return self.layer_quantizers[module.__class__](module, self.cfg)

    def save_quant_params(self, params, name):
        os.makedirs(self.cfg['save_folder'], exist_ok=True)

        with open(osp.join(self.cfg['save_folder'], name + '.json'), 'w') as f:
            json.dump(params, f)

    def collect_stats(self):
        handles = []
        for module in self.model.modules():
            handles.append(module.register_forward_hook(stats_hook))

        dataloader = DataLoader(Folder(self.cfg['data_source'], self.loader, self.transform),
                            batch_size=self.cfg['batch_size'], shuffle=False,
                            num_workers=self.cfg['num_workers'], drop_last=False)
        self.model.eval()

        if self.cfg['use_gpu']:
            self.model.cuda()

        with torch.no_grad():
            for imgs in tqdm(dataloader):
                if self.cfg['use_gpu']:
                    imgs = imgs.cuda()
                _ = self.model(imgs)

        if self.cfg['use_gpu']:
            self.model.cpu()

        for handle in handles:  # delete forward hooks
            handle.remove()
