import torchvision
import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, max_channels = None):
        super().__init__()

        self.layers = self.load_base_architecture()
        self.swap_maxpools_for_upsamples()
        self.initialize_and_swap_direction_of_conv_layers(max_channels)
        self.make_relus_trainable()
        print(self)

    def load_base_architecture(self):
        arch = torchvision.models.vgg19(pretrained=False, progress=True)
        return arch.features[20:None:-1]

    def swap_maxpools_for_upsamples(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.MaxPool2d):
                self.layers[i] = nn.Upsample(scale_factor = (2, 2), \
                                             mode = 'nearest')

    def initialize_and_swap_direction_of_conv_layers(self, max_channels):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                if max_channels is not None:
                    layer.in_channels = min(layer.in_channels, max_channels)
                    layer.out_channels = min(layer.out_channels, max_channels)
                conv2d = nn.Conv2d(layer.out_channels, layer.in_channels, \
                                   kernel_size = layer.kernel_size, \
                                   stride = layer.stride, \
                                   padding = layer.padding, \
                                   padding_mode = 'reflect')
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(conv2d.weight, \
                                                  nonlinearity='relu')
                    torch.nn.init.zeros_(conv2d.bias)
                self.layers[i] = conv2d
            """
            elif isinstance(layer, nn.BatchNorm2d):
                if max_channels is not None:
                    if layer.num_features > max_channels:
                        self.layers[i] = nn.BatchNorm2d(max_channels)
            """

    def make_relus_trainable(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

    def forward(self, x):
        return self.layers(x)

