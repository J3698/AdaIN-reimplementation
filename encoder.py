import torchvision
import torch.nn as nn
import torch


class VGG19Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = self.load_pretrained_blocks()
        self.use_reflection_padding_for_convs()
        self.freeze_weights()


    def load_pretrained_blocks(self):
        layers = torchvision.models.vgg19_bn(pretrained=True, progress=True).features

        block1 = layers[0:3]
        block2 = layers[3:10]
        block3 = layers[10:17]
        block4 = layers[17:30]

        return nn.Sequential(block1, block2, block3, block4)


    def use_reflection_padding_for_convs(self):
        for block in self.layers:
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    layer.padding_mode = 'reflect'


    def freeze_weights(self):
        for i in self.parameters():
            i.requires_grad = False
        super().eval()


    def forward(self, x):
        return self.calc_final_and_intermediate_outputs(x)


    def calc_final_and_intermediate_outputs(self, x):
        outputs = []
        for block in self.feats:
            x = block(x)
            outputs.append(x)
        return outputs


    def train(self):
        raise Exception("Encoder should not be trained")
