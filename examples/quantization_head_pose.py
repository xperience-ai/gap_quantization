import argparse
import json
import math
import os
import re
import shutil

import torch
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

from gap_quantization.models.head_models import HeadClassifier, HeadLookClassifier
from gap_quantization.quantization import ModelQuantizer
from gap_quantization.transforms import QuantizeInput, ToTensorNoNorm
from gap_quantization.utils import load_weights


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trained-model',
                        type=str,
                        default='/media/data/checkpoint_ep28.pth.tar',
                        help="path to trained model stored")
    return parser.parse_args()


def main():
    # provide quantization config
    cfg = {
        "bits": 16,  # number of bits to store weights and activations
        "accum_bits": 32,  # number of bits to store intermediate convolution result
        "signed": True,  # use signed numbers
        "save_folder": "results_head_pose",  # folder to save results
        "data_source": "tests/data",  # folder with images to collect dataset statistics
        "use_gpu": False,  # use GPU for inference
        "batch_size": 1,
        "num_workers": 0,  # number of workers for PyTorch dataloader
        "verbose": True,
        "save_params": True,  # save quantization parameters to the file
        "quantize_forward": True,  # replace usual convs, poolings, ... with GAP-like ones
        "num_input_channels": 1,
        "raw_input": True,
        "double_precision": False  # use double precision convolutions
    }

    # provide transforms that would be applied to images loaded with PIL
    args = argument_parser()

    # model for quantization
    model = HeadLookClassifier()
    load_weights(model, args.trained_model)

    quant_transforms = Compose([
        Resize((128, 128)),
        Grayscale(),
        ToTensorNoNorm(),
    ])

    quantizer = ModelQuantizer(model, cfg, quant_transforms)
    quantizer.quantize_model()

    # now model parameters are quantized
    # if CFG['quantize_forward']=True than we can run inference and emulate GAP8
    # in this case we should quantize input to the network with QuantizeInput(overall_bits, integer_bits)

    quantizer.dump_activations('tests/data/lena.jpg', quant_transforms)

    inp = Image.open('tests/data/lena.jpg')

    with torch.no_grad():
        quant_out = model(quant_transforms(inp).unsqueeze_(0))
    # quant_out is the result of quantize inference
    # quantized output is 2^fraction bits larger than the real output
    # to get the real scale we should divide by the out_frac_bits of the last convolution layer

    rounded_out_1 = quant_out[-1] / math.pow(2., model.look_fc.out_frac_bits)  # pylint: disable=unused-variable
    rounded_out_0 = quant_out[0] / math.pow(2., model.face_fc.out_frac_bits)  # pylint: disable=unused-variable

    remove_cat_files(cfg['save_folder'])


def remove_cat_files(directory):
    for i in [3, 4, 6, 7, 9, 10, 11, 12]:
        os.remove(os.path.join(directory, 'features*.*.' + ".cat.json"))


if __name__ == '__main__':
    main()
