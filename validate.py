from adain import adain
import torch
import matplotlib.pyplot as plt


def validate_style_loss(adain_model, dataloader, epoch_num, writer, device):
    adain_model.eval()

    content_image, style_image = next(iter(dataloader))
    content_image = content_image.to(device)
    style_image = style_image.to(device)

    stylized_sample = adain_model(content_image, style_image)[-1][0]

    style_image = prep_img_for_tensorboard(style_image)
    content_image = prep_img_for_tensorboard(content_image)
    stylized_image = prep_img_for_tensorboard(stylized)

    writer.add_image('style', style_image, epoch_num)
    writer.add_image('content', content_image, epoch_num)
    writer.add_image('stylized', stylized_image, epoch_num)


def prep_img_for_tensorboard(image):
    image_copy = image.cpu().detach().clone()
    return torch.clamp(image_copy, 0, 1)

