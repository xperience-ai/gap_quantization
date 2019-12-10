import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from gap_quantization.quantization import ModelQuantizer

# provide quantization config
CFG = {
    "bits": 16,  # number of bits to store weights and activations
    "accum_bits": 32,  # number of bits to store intermediate convolution result
    "signed": True,  # use signed numbers
    "save_folder": "results",  # folder to save results
    "data_source": "tests/data",  # folder with images to collect dataset statistics
    "use_gpu": False,  # use GPU for inference
    "batch_size": 1,
    "num_workers": 0,  # number of workers for PyTorch dataloader
    "verbose": False,
    "save_params": True,
    "quantize_forward": True,
    "num_input_channels": 3,
    "raw_input": False
}


class CustomToTensor:
    def __call__(self, img):
        return torch.FloatTensor(np.array(img).transpose(2, 0, 1))


class MyModel(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(MyModel, self).__init__()
        self.conv2d = nn.Conv2d(input_ch, output_ch, kernel_size)
        self.conv2d.weight.data = torch.randn(output_ch, input_ch, kernel_size, kernel_size)
        self.conv2d.bias.data = torch.randn(output_ch)

    def forward(self, inp):
        return self.conv2d(inp)


@pytest.fixture
def transforms():
    return CustomToTensor()


@pytest.fixture
def model():
    return MyModel(3, 64, 3)


def test_conv_quant(model, transforms):
    float_weight = model.conv2d.weight
    float_bias = model.conv2d.bias
    quantizer = ModelQuantizer(model, CFG, transforms)
    quantizer.quantize_model()
    rounded_weight = model.conv2d.weight * math.pow(2., -model.conv2d.w_frac_bits)
    rounded_bias = model.conv2d.bias * math.pow(2., -model.conv2d.b_frac_bits)
    assert np.allclose(float_weight.data.cpu().numpy(), rounded_weight.data.cpu().numpy(), atol=1e-2)
    assert np.allclose(float_bias.data.cpu().numpy(), rounded_bias.data.cpu().numpy(), atol=0.2)
