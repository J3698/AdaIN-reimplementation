import torchvision
import torch.nn as nn
import torch


class VGG19Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feats = nn.Sequential(*self.extract_vgg19_pretrained_layers())


    def extract_vgg19_pretrained_layers(self):
        # get pretrained model
        features = torchvision.models.vgg19_bn(pretrained=True, progress=True).features

        # change to reflection
        for i in features:
            if isinstance(i, nn.Conv2d):
                i.padding_mode = 'reflect'

        # get blocks of layers we want
        feats1 = features[0:3]
        feats2 = features[3:10]
        feats3 = features[10:17]
        feats4 = features[17:30]

        return feats1, feats2, feats3, feats4

    def freeze(self):
        for i in self.parameters():
            i.requires_grad = False

    def forward(self, x):
        # get the final output, along with
        # necessary intermediate outputs
        outputs = []
        for block in self.feats:
            x = block(x)
            outputs.append(x)
        return outputs
