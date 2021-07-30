#!/usr/bin/python3

import sys
sys.path.append("..")

import torch.nn as nn
#from encoder import VGG19Encoder
from training.decoder import Decoder
#from adain import adain
from torch.utils.data import DataLoader
#from data import StyleTransferDataset, get_transforms
import torch


def main():
    export_test()

def export_stylizer():
    style = torch.ones((1, 3, 256, 256), requires_grad = False)

    content = torch.ones((1, 3, 256, 256), requires_grad = False)
    model = OneStyleStylizer(style)

    torch.onnx.export(model, content, "exports/stylizer.onnx",
                      export_params = True,
                      opset_version = 11,
                      do_constant_folding = True,
                      input_names = ['input'],
                      output_names = ['output'])

def export_test():
    image = torch.ones((1, 3, 256, 256), requires_grad = False)
    model = Test()

    torch.onnx.export(model, image, "exports/test.onnx",
                      export_params = True,
                      opset_version = 11,
                      do_constant_folding = True,
                      input_names = ['input'],
                      output_names = ['output'])



class Test(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / 2



class OneStyleStylizer(nn.Module):
    def __init__(self, style):
        super().__init__()

        self.encoder = VGG19Encoder()
        for i in self.encoder.parameters():
            i.requires_grad = False

        self.decoder = Decoder()
        for i in self.decoder.parameters():
            i.requires_grad = False

        self.style_features = self.encoder(style)[-1]


    def forward(self, content):
        content_features = self.encoder(content)
        stylized_content_features = adain(content_features[-1], \
                                          self.style_features)
        stylized_image = self.decoder(stylized_content_features)

        return stylized_image



if __name__ == "__main__":
    main()


