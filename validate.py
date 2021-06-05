from adain import adain
import torch
import matplotlib.pyplot as plt


def validate_style_loss(encoder, decoder, dataloader, epoch_num, writer, device):
    global g_batch_size

    encoder.eval()
    decoder.eval()

    content_image, style_image = next(iter(dataloader))
    content_image = content_image.to(device)
    style_image = style_image.to(device)
    g_batch_size, _, w, h = content_image.shape

    stylized = style_transfer(encoder, decoder, content_image, style_image)

    s_image = prep_img_for_tb(style_image[0])
    writer.add_image('style', s_image, epoch_num)
    c_image = prep_img_for_tb(content_image[0])
    writer.add_image('content', c_image, epoch_num)
    f_image = prep_img_for_tb(stylized[0])
    writer.add_image('stylized', f_image, epoch_num)


def show_tensor(tensor, num, run, info = ""):
    if info != "":
        info = "-" + info

    image = tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    image[image > 1] = 1
    image[image < 0] = 0
    plt.imshow(image)
    plt.savefig(f"demo/{run}/{num}{info}.png")


def style_transfer(encoder, decoder, content_image, style_image):
    style_features = encoder(style_image)
    content_features = encoder(content_image)
    stylized_features = adain(content_features[-1], style_features[-1])
    stylized_images = decoder(stylized_features)
    return stylized_images


def prep_img_for_tb(image):
    copy = image.cpu().detach().clone()
    clamp = torch.clamp(copy, 0, 1)

    return clamp
