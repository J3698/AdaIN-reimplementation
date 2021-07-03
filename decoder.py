import torchvision
import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = self.load_base_architecture()
        self.swap_maxpools_for_upsamples(self)
        self.initialize_and_swap_direction_of_conv_layers()
        self.make_relus_trainable()


    def load_base_architecture(self):
        return torchvision.models.vgg19(pretrained=False, progress=True).features[20:None:-1]


    def swap_maxpools_for_upsamples(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.MaxPool2d):
                self.layers[i] = nn.Upsample(scale_factor = (2, 2), mode = 'nearest')


    def initialize_and_swap_direction_of_conv_layers(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                conv2d = nn.Conv2d(layer.out_channels, layer.in_channels, \
                                   kernel_size = layer.kernel_size, stride = layer.stride, \
                                   padding = layer.padding, padding_mode = 'reflect')
                with torch.no_grad():
                    torch.nn.init.kaiming_normal_(conv2d.weight, nonlinearity='relu')
                    torch.nn.init.zeros_(conv2d.bias)
                self.layers[i] = conv2d


    def make_relus_trainable(self)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ReLU):
                layer.inplace = False


    def forward(self, x):
        return self.layers(x)

