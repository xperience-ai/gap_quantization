import torch

from gap_quantization.utils import integerize


class GrayScale:
    def __call__(self, img):
        return img.convert('L')


class ToTensorNoNorm:
    def __call__(self, pic):
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        return img


class QuantizeInput:
    def __init__(self, bits, int_bits):
        self.bits = bits
        self.int_bits = int_bits[0]
        self.float_bits = self.bits - self.int_bits - 1
        self.counter = 0

    def __call__(self, tensor):
        return integerize(tensor, self.float_bits, self.bits)
