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
        'w_frac_bits': [w_frac_bits],
        'b_frac_bits': [bias_frac_bits],
        'inp_frac_bits': [inp_frac_bits]
    }
    params['out_frac_bits'] = [w_frac_bits + inp_frac_bits - params['norm'][0]]
    return params


def concat_quant(module, cfg):
    max_int_bits = max(module.inp_int_bits)
    params = {
        'norm': [max_int_bits - int_bits for int_bits in module.inp_int_bits],
        'out_frac_bits': [cfg['bits'] - max_int_bits - cfg['signed']]
    }
    return params


LAYER_QUANTIZERS = {
    nn.Conv2d: conv_quant,
    Concat: concat_quant,
    EltWiseAdd: concat_quant,
    nn.Linear: conv_quant
}
