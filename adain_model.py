import torch.nn as nn
import torch
from train import create_stylized_images
from adain import adain

class StyleTransferModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = VGG19Encoder()
        self.decoder = Decoder()


    def forward(self, style_images, content_images):
        style_features = self.encoder(style_images)
        content_features = self.encoder(content_images)
        stylized_features, stylized_images = create_stylized_images(content_features, style_features)

        return style_features, content_features, stylized_features, stylized_images


    def create_stylized_images(self, content_features, style_features):
        stylized_features = adain(content_features[-1], style_features[-1])
        stylized_images = self.decoder(stylized_features)

        return stylized_features, stylized_images


    def train(self):
        self.decoder.train()


    def eval(self):
        self.decoder.eval()

