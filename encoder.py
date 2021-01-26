import torchvision
import torch.nn as nn
import torch


def main():
    encoder = VGG19Encoder()
    print(encoder)
    sample_input = torch.ones((1, 3, 256, 256))
    sample_output = encoder(sample_input)
    print(f"Input shapes: {sample_input.shape}")
    print(f"Output shapes: {[i.shape for i in sample_output]}")


class VGG19Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feats = nn.Sequential(*self.extract_vgg19_pretrained_layers())


    def extract_vgg19_pretrained_layers(self):
        features = torchvision.models.vgg19(pretrained=True, progress=True).features

        feats1 = features[0:2]
        feats2 = features[2:7]
        feats3 = features[7:12]
        feats4 = features[12:21]

        return feats1, feats2, feats3, feats4


    def forward(self, x):
        outputs = []
        for block in self.feats:
            x = block(x)
            outputs.append(x)
        return outputs


if __name__ == "__main__":
    main()
