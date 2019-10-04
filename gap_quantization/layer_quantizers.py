import torch.nn as nn

from gap_quantization.layers import Concat, EltWiseAdd
from gap_quantization.utils import int_bits, integerize


def conv_quant(module, cfg):
    out_int_bits = module.out_int_bits

    w_int_bits = int_bits(module.weight)
    w_frac_bits = cfg['bits'] - w_int_bits - cfg['signed']
    assert len(module.inp_int_bits) == 1
    inp_frac_bits = cfg['bits'] - module.inp_int_bits[0] - cfg['signed']

    if out_int_bits + w_frac_bits + inp_frac_bits > cfg['accum_bits'] - cfg['signed']:
        w_frac_bits -= out_int_bits + w_frac_bits + \
            inp_frac_bits - cfg['accum_bits'] + cfg['signed']

    bias_frac_bits = cfg['bits'] - out_int_bits - cfg['signed']
    params = {
        'norm': [max(0, out_int_bits + w_frac_bits + inp_frac_bits - cfg['bits'] + cfg['signed'])],
        'weight': integerize(module.weight.data, w_frac_bits, cfg['bits']).cpu().tolist(),
        'bias': integerize(module.bias.data, bias_frac_bits, cfg['bits']).cpu().tolist(),
        '_w_frac_bits': [w_frac_bits],
        '_b_frac_bits': [bias_frac_bits]
    }
    params['dot_place'] = [w_frac_bits + inp_frac_bits - params['norm'][0]]
    return params


def concat_quant(module, cfg):
    min_int_bits = min(module.int_inp_bits)
    params = {
        'norm': [int_bits - min_int_bits for int_bits in module.int_inp_bits],
        'dot_place': [cfg['bits'] - min_int_bits - cfg['signed']]
    }
    return params


LAYER_QUANTIZERS = {nn.Conv2d: conv_quant, Concat: concat_quant, EltWiseAdd: concat_quant}
