import math

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize

from gap_quantization.models.head_models import HeadLookClassifier, HeadClassifier
from gap_quantization.quantization import ModelQuantizer
from gap_quantization.transforms import ToTensorNoNorm, GrayScale
from gap_quantization.utils import load_weights


GAP_CFG = {
    "bits": 16,  # number of bits to store weights and activations
    "accum_bits": 32,  # number of bits to store intermediate convolution result
    "signed": True,  # use signed numbers
    "save_folder": "results",  # folder to save results
    "data_source": "tests/data",  # folder with images to collect dataset statistics
    "use_gpu": False,  # use GPU for inference
    "batch_size": 1,
    "num_workers": 0,  # number of workers for PyTorch dataloader
    "verbose": False,
    "save_params": False,
    "quantize_forward": True,
    "num_input_channels": 1,
    "raw_input": True,
    "double_precision": False
}


def test_full_head_quant_infer():
    model = HeadLookClassifier()
    # load_weights(model, '/media/slow_drive/head_pose_checkpoints/look_angle_30_degrees_no_norm/checkpoint_ep50.pth.tar')
    model.eval()
    inp = Image.open('tests/data/lena.jpg')

    float_transforms = Compose(
        [Resize((128, 128)),
         GrayScale(),
         ToTensorNoNorm(),
         ])

    with torch.no_grad():
        float_out = model(float_transforms(inp).unsqueeze_(0))

    quantizer = ModelQuantizer(model, GAP_CFG, float_transforms)
    quantizer.quantize_model()

    quantizer.dump_activations('tests/data/lena.jpg', float_transforms)
    with torch.no_grad():
        quant_out = model(float_transforms(inp).unsqueeze_(0))

        rounded_out_1 = quant_out[-1] / math.pow(2., model.look_fc.out_frac_bits)
        rounded_out_0 = quant_out[0] / math.pow(2., model.face_fc.out_frac_bits)

    assert np.allclose(quant_out[0] % 1, 0)
    assert np.allclose(quant_out[1] % 1, 0)
    assert np.allclose(float_out[-1].data.cpu().numpy(), rounded_out_1.data.cpu().numpy(), rtol=0.01)
    assert np.allclose(float_out[0].data.cpu().numpy(), rounded_out_0.data.cpu().numpy(), rtol=0.01)


def test_head_quant_infer():
    model = HeadClassifier()
    #  load_weights(model, '/media/slow_drive/head_pose_checkpoints/look_angle_30_degrees_no_norm/checkpoint_ep50.pth.tar')
    model.eval()
    inp = Image.open('tests/data/lena.jpg')

    float_transforms = Compose(
        [Resize((128, 128)),
         GrayScale(),
         ToTensorNoNorm(),
         ])

    with torch.no_grad():
        float_out = model(float_transforms(inp).unsqueeze_(0))

    quantizer = ModelQuantizer(model, GAP_CFG, float_transforms)
    quantizer.quantize_model()

    quantizer.dump_activations('tests/data/lena.jpg', float_transforms)
    with torch.no_grad():
        quant_out = model(float_transforms(inp).unsqueeze_(0))

        rounded_out = quant_out / math.pow(2., model.face_fc.out_frac_bits)

    np_float_out = float_out.data.cpu().numpy()
    np_rounded_out = rounded_out.data.cpu().numpy()

    assert np.allclose(quant_out % 1, 0)
    assert np.allclose(np_float_out, np_rounded_out, rtol=0.01)
