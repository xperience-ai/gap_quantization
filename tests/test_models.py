import pytest
import torch
from torchvision import models

from gap_quantization.models.mobilenet import mobilenet_v2
# yapf: disable
from gap_quantization.models.resnet import (resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d,
                                            resnext101_32x8d)
# yapf: enable
from gap_quantization.models.squeezenet import squeezenet1_0, squeezenet1_1

MODELS_DICT = {
    'mobilenet_v2': mobilenet_v2,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_1': squeezenet1_1,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
}


def _test_classification_model(name, input_shape):
    # passing num_class equal to a number other than 1000 helps in making the test
    # more enforcing in nature
    model = MODELS_DICT[name](num_classes=50)
    model.eval()
    x = torch.rand(input_shape)
    out = model(x)
    assert out.shape[-1] == 50


def _test_equal(name, input_shape):
    model1 = MODELS_DICT[name](pretrained=True, num_classes=50)
    model1.eval()
    x = torch.rand(input_shape)
    out1 = model1(x)
    model = models.__dict__[name](num_classes=50)
    model.load_state_dict(model1.state_dict())
    model.eval()
    out2 = model(x)
    assert all(out1.detach().numpy().round(6).flatten() == out2.detach().numpy().round(6).flatten())


@pytest.mark.parametrize("model_name", MODELS_DICT.keys())
def test_model(model_name):
    input_shape = (1, 3, 224, 224)
    _test_classification_model(model_name, input_shape)
    _test_equal(model_name, input_shape)
