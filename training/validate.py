from adain import adain
import torch
import matplotlib.pyplot as plt


def validate_style_loss(adain_model, dataloader, epoch_num, writer, device):
    adain_model.eval()

    content_images, style_images = next(iter(dataloader))
    content_images = content_images.to(device)
    style_images = style_images.to(device)

    stylized_sample = adain_model(content_images, style_images)[-1][0]

    style_image = prep_img_for_tensorboard(style_images[0])
    content_image = prep_img_for_tensorboard(content_images[0])
    stylized_sample = prep_img_for_tensorboard(stylized_sample)

    writer.add_image('style', style_image, epoch_num)
    writer.add_image('content', content_image, epoch_num)
    writer.add_image('stylized', stylized_sample, epoch_num)


def prep_img_for_tensorboard(image):
    image_copy = image.cpu().detach().clone()
    return torch.clamp(image_copy, 0, 1)

