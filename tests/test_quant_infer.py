import math

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from gap_quantization.models.squeezenet import squeezenet1_1
from gap_quantization.quantization import ModelQuantizer
from gap_quantization.transforms import QuantizeInput

# provide quantization config
CFG = {
    "bits": 16,  # number of bits to store weights and activations
    "accum_bits": 32,  # number of bits to store intermediate convolution result
    "signed": True,  # use signed numbers
    "save_folder": "results",  # folder to save results
    "data_source": "tests/data",  # folder with images to collect dataset statistics
    "batch_size": 1,
    "num_workers": 0,  # number of workers for PyTorch dataloader
    "verbose": False,
    "save_params": False,
    "quantize_forward": True,
    "num_input_channels": 3
}


@pytest.fixture
def squeezenet():
    return squeezenet1_1(pretrained=True, progress=False)


def test_squeezenet_infer(squeezenet):
    model = squeezenet
    model.eval()
    inp = Image.open('tests/data/lena.jpg')

    float_transforms = Compose(
        [Resize((128, 128)),
         ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        float_out = model(float_transforms(inp).unsqueeze_(0))

    quantizer = ModelQuantizer(model, CFG, float_transforms)
    quantizer.quantize_model()

    quant_transforms = Compose([
        Resize((128, 128)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        QuantizeInput(CFG['bits'],
                      next(model.modules()).inp_int_bits)
    ])

    with torch.no_grad():
        quant_out = model(quant_transforms(inp).unsqueeze_(0))
        for layer in reversed(list(model.modules())):
            if hasattr(layer, 'out_frac_bits'):
                out_frac_bits = layer.out_frac_bits
                break
        rounded_out = quant_out / math.pow(2., out_frac_bits)

    np_float_out = float_out.data.cpu().numpy()
    np_rounded_out = rounded_out.data.cpu().numpy()

    assert np.allclose(quant_out % 1, 0)
    assert np.allclose(np_float_out, np_rounded_out, atol=0.5, rtol=0.1)
