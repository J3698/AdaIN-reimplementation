import torchvision
import torch.nn as nn
import torch
from encoder import VGG19Encoder


def main():
    encoder = VGG19Encoder()
    decoder = Decoder()
    print(decoder)

    sample_input = torch.ones((1, 3, 256, 256))
    outputs = encoder(sample_input)
    output = decoder(outputs[-1])

    print(f"Input shape: {sample_input.shape}")
    print(f"Intermediate shapes: {[output.shape for output in outputs]}")
    print(f"Output shape: {output.shape}")


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = torchvision.models.vgg19(pretrained=False, progress=True).features[20:None:-1]
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                self.features[i] = nn.Upsample(scale_factor = (2, 2), mode = 'nearest')
            elif isinstance(layer, nn.Conv2d):
                conv2d = nn.Conv2d(layer.out_channels, layer.in_channels, \
                                   kernel_size = layer.kernel_size, stride = layer.stride, \
                                   padding = layer.padding, padding_mode = 'reflect')
                self.features[i] = conv2d


    def forward(self, x):
        return self.features(x)


if __name__ == "__main__":
    main()
