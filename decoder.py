import torchvision
import torch.nn as nn
import torch
from encoder import VGG19Encoder


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        features = torchvision.models.vgg19(pretrained=True, progress=True).features[20:None:-1]
        for i, layer in enumerate(features):
            if isinstance(layer, nn.MaxPool2d):
                features[i] = nn.Upsample(scale_factor = (2, 2), mode = 'nearest')
            elif isinstance(layer, nn.Conv2d):
                conv2d = nn.Conv2d(layer.out_channels, layer.in_channels, \
                                   kernel_size = layer.kernel_size, stride = layer.stride, \
                                   padding = layer.padding, padding_mode = 'reflect')
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(conv2d.weight, nonlinearity='relu')
                    torch.nn.init.zeros_(conv2d.bias)
                features[i] = conv2d
            elif isinstance(layer, nn.ReLU):
                layer.inplace = False

        # features[-1].bias = None
        self.features = features


    def forward(self, x):
        out = self.features(x)
        return out
