import math

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from gap_quantization.models.squeezenet import squeezenet1_1
from gap_quantization.quantization import ModelQuantizer
from gap_quantization.transforms import QuantizeInput


def main():
    # provide quantization config
    cfg = {
        "bits": 16,  # number of bits to store weights and activations
        "accum_bits": 32,  # number of bits to store intermediate convolution result
        "signed": True,  # use signed numbers
        "save_folder": "results",  # folder to save results
        "data_source": "tests/data",  # folder with images to collect dataset statistics
        "use_gpu": True,  # use GPU for inference
        "batch_size": 1,
        "num_workers": 0,  # number of workers for PyTorch dataloader
        "verbose": False,
        "save_params": True,  # save quantization parameters to the file
        "quantize_forward": True  # replace usual convs, poolings, ... with GAP-like ones
    }

    # provide transforms that would be applied to images loaded with PIL
    transforms = Compose(
        [Resize((128, 128)),
         ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # model for quantization
    model = squeezenet1_1(pretrained=True, progress=False)

    quantizer = ModelQuantizer(model, cfg, transforms)
    quantizer.quantize_model()

    # now model parameters are quantized
    # if CFG['quantize_forward']=True than we can run inference and emulate GAP8
    # in this case we should quantize input to the network with QuantizeInput(overall_bits, integer_bits)

    quant_transforms = Compose([
        Resize((128, 128)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        QuantizeInput(cfg['bits'],
                      next(model.modules()).inp_int_bits)
    ])

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


if __name__ == '__main__':
    main()
