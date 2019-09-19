import torch.nn as nn

from gap_quantization.utils import integerize, int_bits
from gap_quantization.layers import Concat, EltWiseAdd


def conv_quant(module, cfg):
    out_int_bits = module.out_int_bits

    w_int_bits = int_bits(module.weight)
    w_frac_bits = cfg['bits'] - w_int_bits - cfg['signed']
    assert len(module.inp_int_bits) == 1
    inp_frac_bits = cfg['bits'] - module.inp_int_bits[0] - cfg['signed']

    if out_int_bits + w_frac_bits + inp_frac_bits > cfg['accum_bits'] - cfg['signed']:
        w_frac_bits -= out_int_bits + w_frac_bits + \
            inp_frac_bits - cfg['accum_bits'] + cfg['signed']

    params = {'norm': max(0, out_int_bits + w_frac_bits + inp_frac_bits - cfg['bits'] + cfg['signed']),
              'weight': integerize(module.weight.data, w_frac_bits, cfg['bits']).cpu().tolist(),
              'bias': integerize(module.bias.data, cfg['bits'] - out_int_bits - cfg['signed'],
                                 cfg['bits']).cpu().tolist()}
    params['dot_place'] = w_frac_bits + inp_frac_bits - params['norm']
    return params


def concat_quant(module, cfg):
    min_int_bits = min(module.int_inp_bits)
    params = {'norm': [int_bits - min_int_bits for int_bits in module.int_inp_bits],
              'dot_place': cfg['bits'] - min_int_bits - cfg['signed']}
    return params


LAYER_QUANTIZERS = {nn.Conv2d: conv_quant,
                    Concat: concat_quant,
                    EltWiseAdd: concat_quant
                    }
