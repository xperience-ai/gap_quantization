import torch.nn as nn
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from gap_quantization.quantization import ModelQuantizer

# provide quantization config
CFG = {
    "bits": 16,  # number of bits to store weights and activations
    "accum_bits": 32,  # number of bits to store intermediate convolution result
    "signed": True,  # use signed numbers
    "save_folder": "results",  # folder to save results
    "data_source": "data",  # folder with images to collect dataset statistics
    "use_gpu": True,  # use GPU for inference
    "batch_size": 1,
    "num_workers": 0,  # number of workers for PyTorch dataloader
    "verbose": False
}

# provide transforms that would be applied to images loaded with PIL
TRANSFORMS = Compose(
    [Resize((128, 128)),
     ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class MyModel(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size):
        super(MyModel, self).__init__()
        self.conv2d = nn.Conv2d(input_ch, output_ch, kernel_size)

    def forward(self, inp):
        return self.conv2d(inp)


# model for quantization
MODEL = MyModel(3, 64, 3)

QUANTIZER = ModelQuantizer(MODEL, CFG, TRANSFORMS)
QUANTIZER.quantize_model()
