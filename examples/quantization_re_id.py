import argparse
import json
import math
import os
import re
import shutil

import torch
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

from gap_quantization.dump_utils import dump_quant_params, remove_cat_files, remove_extra_dump
from gap_quantization.models.squeezenet1_1 import squeezenet1_1
from gap_quantization.quantization import ModelQuantizer
from gap_quantization.transforms import QuantizeInput, ToTensorNoNorm


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trained-model', type=str, default='test', help="path to trained model stored")
    parser.add_argument('--convbn',
                        type=str,
                        default=True,
                        help='Normalize features vector on the output of the network')
    return parser.parse_args()


def main():
    # provide quantization config
    cfg = {
        "bits": 16,  # number of bits to store weights and activations
        "accum_bits": 32,  # number of bits to store intermediate convolution result
        "signed": True,  # use signed numbers
        "save_folder": "results",  # folder to save results
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
    transforms = Compose([Resize((128, 128)), Grayscale(), ToTensorNoNorm()])
    args = argument_parser()
    convbn = args.convbn

    # model for quantization
    model = squeezenet1_1(num_classes=8631,
                          loss={'xent', 'htri'},
                          pretrained=False,
                          grayscale=True,
                          normalize_embeddings=False,
                          normalize_fc=False,
                          convbn=convbn)

    save_path = args.trained_model
    pretrained_dict = torch.load(save_path)['state_dict']
    model.load_state_dict(pretrained_dict)

    quantizer = ModelQuantizer(model, cfg, transforms)
    quantizer.quantize_model()

    # now model parameters are quantized
    # if CFG['quantize_forward']=True than we can run inference and emulate GAP8
    # in this case we should quantize input to the network with QuantizeInput(overall_bits, integer_bits)

    quant_transforms = Compose([
        Resize((128, 128)),
        Grayscale(),
        ToTensorNoNorm(),
    ])

    quantizer.dump_activations('tests/data/lena.jpg', quant_transforms)

    inp = Image.open('tests/data/lena.jpg')

    with torch.no_grad():
        quant_out = model(quant_transforms(inp).unsqueeze_(0))
    # quant_out is the result of quantize inference
    # quantized output is 2^fraction bits larger than the real output
    # to get the real scale we should divide by the out_frac_bits of the last convolution layer
    for layer in reversed(list(model.modules())):
        if hasattr(layer, 'out_frac_bits'):
            out_frac_bits = layer.out_frac_bits
            break
    rounded_out = quant_out / math.pow(2., out_frac_bits)  # pylint: disable=unused-variable
    print(rounded_out)

    dump_quant_params(cfg['save_folder'], convbn)
    if convbn:
        remove_extra_dump(os.path.join(cfg['save_folder'], 'activations_dump'))
    remove_cat_files(cfg['save_folder'])


if __name__ == '__main__':
    main()
