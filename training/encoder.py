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
        for block in self.blocks:
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    layer.padding_mode = 'reflect'

    def freeze_weights(self):
        for i in self.parameters():
            i.requires_grad = False
        self.eval()

    def forward(self, x):
        return self.calc_final_and_intermediate_outputs(x)

    def calc_final_and_intermediate_outputs(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs

    def train(self, is_train = None):
        if is_train is None or is_train:
            raise Exception("Encoder should not be trained")
        super().train(is_train)


class Resnet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.blocks = self.load_pretrained_blocks()
        self.use_reflection_padding_for_convs()
        self.freeze_weights()
        self.set_relu_out_of_place()

    def load_pretrained_blocks(self):
        net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', \
                                pretrained=True)

        block1 = nn.Sequential(net.conv1, net.bn1, net.relu)
        block2 = net.layer1[0]
        block3 = net.layer1[1]
        block4 = nn.Sequential(net.layer2, nn.AvgPool2d((2, 2)))

        return nn.Sequential(block1, block2, block3, block4)

    def use_reflection_padding_for_convs(self):
        for name, module in self.blocks.named_modules():
            if isinstance(module, nn.Conv2d):
                module.padding_mode = 'reflect'

    def set_relu_out_of_place(self):
        for name, module in self.blocks.named_modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False


    def freeze_weights(self):
        for i in self.parameters():
            i.requires_grad = False
        self.eval()

    def forward(self, x):
        return self.calc_final_and_intermediate_outputs(x)

    def calc_final_and_intermediate_outputs(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs

    def train(self, is_train = None):
        if is_train is None or is_train:
            raise Exception("Encoder should not be trained")
        super().train(is_train)

if __name__ == "__main__":
    enc1 = Resnet18Encoder()
    enc2 = VGG19Encoder()
    breakpoint()


