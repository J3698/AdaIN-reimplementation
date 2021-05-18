from encoder import VGG19Encoder
from decoder import Decoder
import os
from adain import adain
from math import ceil
import tqdm
from torch.utils.data import DataLoader
from data import StyleTransferDataset, IterableStyleTransferDataset, get_transforms
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F
import time
import random
from torch.utils.tensorboard import SummaryWriter
import argparse


def train(encoder, decoder, dataloader, val_dataloader, optimizer, scheduler, args, writer, saved_epoch, run, device):
    loss = None
    for epoch in range(saved_epoch, args.num_epochs):
        if args.save_freq != 0 and epoch % args.save_freq == 0 and False:
            torch.save({
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch, 'loss': loss,
                'num_workers': args.num_workers, 'device': device,
                'batch_size': args.batch_size, 'dataset_length': args.dataset_length,
                'num_epochs': args.num_epochs, 'lr': args.lr,
                'lambda_style': args.lambda_style, 'lambda_content': args.lambda_content,
                'seed': args.seed, 'coco_path': args.coco_path,
                'coco_labels_path': args.coco_labels_path, 'wiki_path': args.wikiart_path,
            }, f"demo/{run}/{epoch}.pt")

        loss = train_epoch_style_loss(args, encoder, decoder, dataloader, \
                                      val_dataloader, optimizer,\
                                      epoch, writer, run, device)
        scheduler.step()


def train_epoch_style_loss(args, encoder, decoder, dataloader,\
                           val_dataloader, optimizer, \
                           epoch_num, writer, run, device):
    global g_batch_size

    encoder.train()
    decoder.train()

    total_loss = 0
    num_iters = len(dataloader)
    if isinstance(dataloader.dataset, IterableStyleTransferDataset):
        num_iters = ceil(len(dataloader) / args.batch_size)
        print(num_iters)
    print(num_iters)

    progress_bar = tqdm.tqdm(enumerate(dataloader),\
                             total = num_iters, dynamic_ncols = True)
    for i, (content_image, style_image) in progress_bar:

        content_image = content_image.to(device)
        style_image = style_image.to(device)
        g_batch_size, _, w, h = content_image.shape
        # assert content_image.shape == style_image.shape

        optimizer.zero_grad()
        style_loss, content_loss, stylized = \
                get_style_transfer_loss(encoder, decoder, \
                                        content_image, style_image,\
                                        args.lambda_content, args.lambda_style)
        full_loss = style_loss + content_loss
        full_loss.backward()

        total_loss += full_loss.item()

        progress_bar.set_postfix({'epoch': f"{epoch_num}",
                                  'loss': f"{total_loss / (i + 1):.2f}"})

        optimizer.step()

        it = epoch_num * num_iters + i
        #if i % 500 == 0:
        if epoch_num % 20 == 0:
            imgs_per_epoch = num_iters // 500
            img_num = imgs_per_epoch * epoch_num + i // 500
            validate_style_loss(encoder, decoder, val_dataloader,\
                                epoch_num, writer, device)

            encoder.train()
            decoder.train()

        writer.add_scalar('SLoss/train_it', style_loss.item(), it)
        writer.add_scalar('CLoss/train_it', content_loss.item(), it)
        writer.add_scalar('Loss/train_it', full_loss.item(), it)



    writer.add_scalar('Loss/train', total_loss, epoch_num)
    return total_loss

def prep_img_for_tb(image):
    copy = image.cpu().detach().clone()
    clamp = torch.clamp(copy, 0, 1)

    return clamp


def get_style_transfer_loss(encoder, decoder, content_image, style_image,\
                            lambda_content, lambda_style):
    assert_shape(content_image, (g_batch_size, 3, 256, 256))

    style_features = encoder(style_image)
    content_features = encoder(content_image)

    stylized_images, stylized_features =\
                create_stylized_images(decoder, content_features, style_features)

    features_of_stylized = encoder(stylized_images)

    style_loss = compute_style_loss(features_of_stylized, style_features)
    content_loss = compute_content_loss(features_of_stylized[-1],\
                                        stylized_features)

    return style_loss * lambda_style, content_loss * lambda_content,\
                                     stylized_images


def style_transfer(encoder, decoder, content_image, style_image):
    style_features = encoder(style_image)
    content_features = encoder(content_image)
    stylized_features = adain(content_features[-1], style_features[-1])
    stylized_images = decoder(stylized_features)
    return stylized_images


def create_stylized_images(decoder, content_features, style_features):
    assert len(content_features) == 4
    assert len(style_features) == 4

    shapes = [i.shape for i in content_features]
    assert shapes == [(g_batch_size, 64, 256, 256),
                      (g_batch_size, 128, 128, 128),
                      (g_batch_size, 256, 64, 64),
                      (g_batch_size, 512, 32, 32) ], shapes
    shapes = [i.shape for i in style_features]
    assert shapes == [(g_batch_size, 64, 256, 256),
                      (g_batch_size, 128, 128, 128),
                      (g_batch_size, 256, 64, 64),
                      (g_batch_size, 512, 32, 32) ], shapes

    stylized_features = adain(content_features[-1], style_features[-1])
    assert stylized_features.shape == (g_batch_size, 512, 32, 32), stylized_features.shape

    stylized_images = decoder(stylized_features)
    assert stylized_images.shape == (g_batch_size, 3, 256, 256)

    return stylized_images, stylized_features


def compute_content_loss(features_of_stylized, stylized_features):
    assert features_of_stylized.shape == (g_batch_size, 512, 32, 32)
    assert stylized_features.shape == (g_batch_size, 512, 32, 32)

    batch_size = features_of_stylized.shape[0]

    content_loss = \
        F.mse_loss(features_of_stylized, stylized_features, reduction = "none")
    assert content_loss.shape == (g_batch_size, 512, 32, 32), content_loss.shape

    content_loss = content_loss.view(batch_size, -1)
    assert content_loss.shape == (batch_size, 512 * 32 * 32)

    content_loss = content_loss.sum(-1) ** 0.5
    assert content_loss.shape == (batch_size,)
    return content_loss.mean()


def compute_style_loss(features_of_stylized, style_features):
    style_loss = 0

    shapes = [i.shape for i in features_of_stylized]
    assert shapes == [(g_batch_size, 64, 256, 256),
                      (g_batch_size, 128, 128, 128),
                      (g_batch_size, 256, 64, 64),
                      (g_batch_size, 512, 32, 32) ], shapes

    shapes = [i.shape for i in style_features]
    assert shapes == [(g_batch_size, 64, 256, 256),
                      (g_batch_size, 128, 128, 128),
                      (g_batch_size, 256, 64, 64),
                      (g_batch_size, 512, 32, 32) ], shapes

    zipped_features = zip(features_of_stylized, style_features)
    for feat_of_stylized, style_feat in zipped_features:
        feature_maps = feat_of_stylized.shape[1]
        assert feat_of_stylized.shape == (g_batch_size, feature_maps,\
                                          2 ** 14 / feature_maps, 2 ** 14 / feature_maps)
        assert style_feat.shape == (g_batch_size, feature_maps,\
                                    2 ** 14 / feature_maps, 2 ** 14 / feature_maps)

        stdevs1, means1 = calc_feature_stats_vectors(feat_of_stylized)
        stdevs2, means2 = calc_feature_stats_vectors(style_feat)
        assert stdevs1.shape == (g_batch_size, feature_maps)
        assert stdevs2.shape == (g_batch_size, feature_maps)
        assert means1.shape == (g_batch_size, feature_maps)
        assert means2.shape == (g_batch_size, feature_maps)

        stdev_loss_vector = F.mse_loss(stdevs1, stdevs2, reduction = "none") ** 0.5
        mean_loss_vector = F.mse_loss(means1, means2, reduction = "none") ** 0.5
        assert stdev_loss_vector.shape == (g_batch_size, feature_maps)
        assert mean_loss_vector.shape == (g_batch_size, feature_maps)

        style_loss += mean_loss_vector.sum(-1).mean()
        style_loss += stdev_loss_vector.sum(-1).mean()

    return style_loss


def calc_feature_stats_vectors(features):
    assert len(features.shape) == 4
    batch_size, feature_maps, w, h = features.shape
    assert batch_size == g_batch_size

    features = features.view(batch_size * feature_maps, -1)
    assert features.shape == (g_batch_size * feature_maps, w * h)

    feature_stdevs = features.var(-1, unbiased = False) ** 0.5
    feature_means = features.mean(-1)
    assert feature_stdevs.shape == (batch_size * feature_maps,), feature_vars.shape
    assert feature_means.shape == (batch_size * feature_maps,), feature_means.shape

    feature_stdevs = feature_stdevs.reshape(batch_size, feature_maps)
    feature_means = feature_means.reshape(batch_size, feature_maps)

    return feature_stdevs, feature_means


def show_tensor(tensor, num, run, info = ""):
    if info != "":
        info = "-" + info

    image = tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    image[image > 1] = 1
    image[image < 0] = 0
    plt.imshow(image)
    plt.savefig(f"demo/{run}/{num}{info}.png")

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


def assert_shape(tensor, expected):
    msg = f"expected {expected} actual {tensor.shape}"
    assert tensor.shape == expected, msg


if __name__ == "__main__":
    main()
