import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.quantization import ModelQuantizer

# provide quantization config
cfg = {"bits": 16,  # number of bits to store weights and activations
       "accum_bits": 32,  # number of bits to store intermediate convolution result
       "signed": True,  # use signed numbers
       "save_folder": "results",  # folder to save results
       "data_source": "data",  # folder with images to collect dataset statistics
       "use_gpu": True,  # use GPU for inference
       "batch_size": 1,
       "num_workers": 0  # number of workers for PyTorch dataloader
       }

# provide transforms that would be applied to images loaded with PIL
transforms = Compose([Resize((128, 128)),
                      ToTensor(),
                      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# model for quantization
model = nn.Sequential(nn.Conv2d(3, 64, 3))

quantizer = ModelQuantizer(model, cfg, transforms)
quantizer.quantize_model()