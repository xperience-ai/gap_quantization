[![Build Status](https://travis-ci.com/xperience-ai/gap_quantization.svg?branch=master)](https://travis-ci.com/xperience-ai/gap_quantization)

# Intro

Placeholder for GAP8 export and quantization module for PyTorch

# Install

```
    $ sudo pip3 install git+https://github.com/xperience-ai/gap_quantization.git
```

## Installation for development

1. Clone the repository from github.
2. Install the development environment:

```
# create virtual environment using virtualenv (in this example)/pyvenv/conda/etc.
$ virtualenv -p /usr/bin/python3 ./venv
$ . venv/bin/activate
$ pip3 install -r requirements-dev.txt
```
3. Instal pre-commit hooks:
```
$ pre-commit install
```

After that, all you commit swill be checked with a set of linting tools (some of them are even able to fix the issues!)
To run these checks independently from the commit process, just type:
```
pre-commit run --all-files
```

# Using
## Supported models import
```
from gap_quantization.models.mobilenet import mobilenet_v2
from gap_quantization.models.squeezenet import squeezenet1_0, squeezenet1_1
from gap_quantization.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
```
## Model initialization
All models are identical to pytorch models. Pretrained weights are loaded from model_zoo.
If last fc layer has different shape, it loads patitialy.
```
model = mobilenet_v2(pretrained=True, num_classes=2)
```
