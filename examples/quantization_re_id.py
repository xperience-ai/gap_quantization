import argparse
import json
import math
import os
import re
import shutil

import torch
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Normalize, Resize, ToTensor

from gap_quantization.models.squeezenet1_1 import squeezenet1_1
from gap_quantization.quantization import ModelQuantizer
from gap_quantization.transforms import QuantizeInput, ToTensorNoNorm


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trained-model', type=str, default='test', help="path to trained model stored")
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
        "num_input_channels": 1
    }

    # provide transforms that would be applied to images loaded with PIL
    transforms = Compose([Resize((128, 128)), Grayscale(), ToTensorNoNorm()])

    # model for quantization
    model = squeezenet1_1(num_classes=8631,
                          loss={'xent', 'htri'},
                          pretrained=False,
                          grayscale=True,
                          normalize_embeddings=False,
                          normalize_fc=False,
                          convbn=True)

    save_path = argument_parser().trained_model
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

    dict_norm = {}
    for file in os.listdir(cfg['save_folder']):
        if file != 'activation_dump' and not re.match('.*cat.json', file):
            dict_norm[file] = round(read_norm(os.path.join(cfg['save_folder'], file)))

    list_norm = make_list_from_dict(dict_norm)
    txt_list = open('norm_list.txt', 'w')
    for i in range(0, 26):
        txt_list.write("#define NORM_" + str(i) + " " + str(list_norm[i]) + "\n")

    remove_extra_dump(os.path.join(cfg['save_folder'], 'activation_dump'))


def read_norm(file):
    with open(file, "rt") as js_file:
        data = json.load(js_file)
    norm = data['norm'][0]
    return norm


def make_list_from_dict(norm_dict):
    norm_list = []
    norm_list.extend([norm_dict['conv1.0.json'], norm_dict['features.0.0.json']])
    for i in [3, 4, 6, 7, 9, 10, 11, 12]:
        norm_list.extend([
            norm_dict['features.' + str(i) + '.squeeze.0.json'],
            norm_dict['features.' + str(i) + '.expand1x1.0.json'],
            norm_dict['features.' + str(i) + '.expand3x3.0.json']
        ])
    return norm_list


def remove_extra_dump(directory):
    extra_list = ['conv1', 'conv1.1', 'features.0', 'features.0.1']
    for i in [3, 4, 6, 7, 9, 10, 11, 12]:
        extra_list.append('features.' + str(i) + '.squeeze')
        extra_list.append('features.' + str(i) + '.squeeze.1')
        extra_list.append('features.' + str(i) + '.expand1x1')
        extra_list.append('features.' + str(i) + '.expand1x1.1')
        extra_list.append('features.' + str(i) + '.expand3x3')
        extra_list.append('features.' + str(i) + '.expand3x3.1')
    for folder in os.listdir(directory):
        if folder in extra_list:
            shutil.rmtree(os.path.join(directory, folder))


if __name__ == '__main__':
    main()
