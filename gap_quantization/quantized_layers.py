import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gap_quantization.layers import Concat, EltWiseAdd
from gap_quantization.utils import gap8_clip, roundnorm_reg


class QuantizedAvgPool2d(nn.AvgPool2d):  # Avg Pooling was tested only like Global Average Pooling
    def __init__(self,
                 kernel_size,
                 stride=None,
                 padding=0,
                 ceil_mode=False,
                 count_include_pad=True,
                 bits=16,
                 **kwargs):  # pylint: disable=unused-argument
        super(QuantizedAvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.bits = bits

    def forward(self, inputs):
        inputs = F.avg_pool2d(inputs, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                              self.count_include_pad)
        inputs = torch.floor_(inputs * self.kernel_size * self.kernel_size + 0.1)
        pool_factor = math.pow(2, 16) // math.pow(self.kernel_size, 2)
        bound = math.pow(2.0, self.bits - 1)
        min_val = -bound
        max_val = bound - 1
        return torch.clamp(roundnorm_reg(inputs * pool_factor, self.bits), min_val, max_val)


class QuantizedAdaptiveAvgPool2d(
        nn.AdaptiveAvgPool2d):  # Avg Pooling was tested only like Global Average Pooling
    def __init__(self, output_size=1, bits=16, **kwargs):  # pylint: disable=unused-argument
        super(QuantizedAdaptiveAvgPool2d, self).__init__(output_size)
        self.bits = bits

    def forward(self, inputs):
        inputs = F.adaptive_avg_pool2d(inputs, self.output_size)
        if isinstance(self.output_size, tuple):
            mult = inputs.shape[2] * inputs.shape[3] // self.output_size[0] // self.output_size[1]
        else:
            mult = inputs.shape[2] * inputs.shape[3] // self.output_size
        inputs = torch.floor_(inputs * mult + 0.1)
        pool_factor = math.pow(2, 16) // mult
        bound = math.pow(2.0, self.bits - 1)
        min_val = -bound
        max_val = bound - 1
        return torch.clamp(roundnorm_reg(inputs * pool_factor, self.bits), min_val, max_val)


class QuantizedConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 bits=16,
                 **kwargs):  # pylint: disable=unused-argument
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                              dilation, groups, bias)
        self.bits = bits

    def forward(self, inputs):
        self.weights = nn.ParameterList(  # pylint: disable=attribute-defined-outside-init
            [nn.Parameter(self.weight.data[:, i, :, :].unsqueeze_(1)) for i in range(self.weight.shape[1])])
        out = None
        for i in range(inputs.shape[1]):
            conv_res = F.conv2d(inputs[:, i, :, :].unsqueeze_(1), self.weights[i], None, self.stride,
                                self.padding, self.dilation, self.groups)
            tmp = gap8_clip(roundnorm_reg(conv_res, self.norm), self.bits)
            if out is None:
                out = tmp
            else:
                out += tmp
        out += self.bias.view(1, -1, 1, 1).expand_as(out)
        out = torch.clamp(out, -math.pow(2., self.bits - 1), math.pow(2., self.bits) - 1)
        return out


class QuantizedConcat(Concat):
    def __init__(self, dim=1, **kwargs):  # pylint: disable=unused-argument
        super(QuantizedConcat, self).__init__(dim)

    def forward(self, inputs):
        for idx, _ in enumerate(inputs):
            inputs[idx] = roundnorm_reg(inputs[idx], self.norm[idx])
        return torch.cat(inputs, self.dim)


class QuantizedEltWiseAdd(EltWiseAdd):
    def __init__(self, **kwargs):  # pylint: disable=unused-argument
        super(QuantizedEltWiseAdd, self).__init__()

    def forward(self, inp1, inp2):
        inputs = [inp1, inp2]
        for idx, _ in enumerate([inp1, inp2]):
            inputs[idx] = roundnorm_reg(inputs[idx], self.norm[idx])
        return torch.stack(inputs, dim=0).sum(dim=0)


QUANTIZED_LAYERS = {
    nn.Conv2d: QuantizedConv2d,
    Concat: QuantizedConcat,
    EltWiseAdd: QuantizedEltWiseAdd,
    nn.AvgPool2d: QuantizedAvgPool2d,
    nn.AdaptiveAvgPool2d: QuantizedAdaptiveAvgPool2d,
    nn.BatchNorm2d: nn.Identity
}
