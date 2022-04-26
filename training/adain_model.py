import torch.nn as nn
import torch
from adain import adain
from encoder import VGG19Encoder, Resnet18Encoder
from decoder import Decoder


class StyleTransferModel(nn.Module):
    def __init__(self, encoder = 'vgg'):
        super().__init__()

        if encoder == 'vgg':
            self.encoder = VGG19Encoder()
            self.decoder = Decoder()
        elif encoder == 'resnet':
            self.encoder = Resnet18Encoder()
            self.decoder = Decoder(128)
        else:
            raise NotImplementedError(f"{encoder} encoder not implemeted.")



    def forward(self, content_images, style_images):
        style_features = self.encoder(style_images)
        content_features = self.encoder(content_images)
        stylized_features, stylized_images = self.create_stylized_images(content_features, style_features)

        return style_features, content_features, stylized_features, stylized_images

    def stylize(self, content_image, style_image):
        style_features = self.encoder(style_images)
        content_features = self.encoder(content_images)
        stylized_features = adain(content_features[-1], style_features[-1])
        stylized_images = self.decoder(stylized_features)

        return stylized_images[0]

    def create_stylized_images(self, content_features, style_features):
        stylized_features = adain(content_features[-1], style_features[-1])
        stylized_images = self.decoder(stylized_features)

        return stylized_features, stylized_images

    def train(self):
        self.decoder.train()

    def eval(self):
        self.decoder.eval()

