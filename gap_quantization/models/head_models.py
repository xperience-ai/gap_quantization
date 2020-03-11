import torch.nn as nn

from gap_quantization.layers import Concat, Flatten


def conv2d(input_channels, out_channels, convbn=False, **kwargs):
    if convbn:
        return nn.Sequential(nn.Conv2d(input_channels, out_channels, **kwargs), nn.BatchNorm2d(out_channels))
    return nn.Conv2d(input_channels, out_channels, **kwargs)


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, convbn, pool=False, ceil_mode=True):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = conv2d(inplanes, squeeze_planes, convbn, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = conv2d(squeeze_planes, expand1x1_planes, convbn, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = conv2d(squeeze_planes, expand3x3_planes, convbn, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

        self.cat = Concat(1)
        self.pool = pool

        if self.pool == 'max':
            self.maxpool1x1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode)
            self.maxpool3x3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=ceil_mode)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        if self.pool == 'max':
            return self.cat([
                self.maxpool1x1(self.expand1x1_activation(self.expand1x1(x))),
                self.maxpool3x3(self.expand3x3_activation(self.expand3x3(x)))
            ])
        else:
            return self.cat([
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x))
            ])


class HeadLookClassifier(nn.Module):
    def __init__(self, convbn=False):
        super(HeadLookClassifier, self).__init__()
        modules1 = [nn.Conv2d(1, 64, kernel_size=3, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(64, 16, 64, 64, convbn),
                    Fire(128, 16, 64, 64, convbn, pool='max', ceil_mode=True)]
        modules2 = [nn.Identity(),
                    Fire(128, 32, 128, 128, convbn),
                    Fire(256, 32, 128, 128, convbn, pool='max', ceil_mode=True),
                    nn.Identity(),
                    Fire(256, 48, 192, 192, convbn),
                    Fire(384, 48, 192, 192, convbn),
                    Fire(384, 64, 256, 256, convbn),
                    Fire(512, 64, 256, 256, convbn)]

        self.features1 = nn.Sequential(*modules1)
        self.features2 = nn.Sequential(*modules2)

        self.face_fc = nn.Linear(128, 1)
        self.face_pool = nn.AdaptiveAvgPool2d(1)
        self.look_fc = nn.Linear(512, 1)
        self.look_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten1 = Flatten(1)
        self.flatten2 = Flatten(1)

    def forward(self, x):
        x = self.features1(x)
        face_vector = self.face_pool(x)
        face_vector = self.flatten1(face_vector)

        face_val = self.face_fc(face_vector)

        x = self.features2(x)
        look_vector = self.look_pool(x)
        look_vector = self.flatten2(look_vector)

        look_val = self.look_fc(look_vector)
        return face_val, look_val


class HeadClassifier(nn.Module):
    def __init__(self, convbn=False):
        super(HeadClassifier, self).__init__()
        modules1 = [nn.Conv2d(1, 64, kernel_size=3, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(64, 16, 64, 64, convbn),
                    Fire(128, 16, 64, 64, convbn, pool='max', ceil_mode=True)]

        self.features1 = nn.Sequential(*modules1)

        self.face_fc = nn.Linear(128, 1)
        self.face_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = Flatten(1)

    def forward(self, x):
        x = self.features1(x)
        face_vector = self.face_pool(x)
        face_vector = self.flatten(face_vector)
        face_logit = self.face_fc(face_vector)  # sigmoid fn is applied usually
        return face_logit
