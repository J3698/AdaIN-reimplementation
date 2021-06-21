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
from validate import validate_style_loss


def train(encoder, decoder, dataloader, val_dataloader, optimizer, \
          scheduler, args, writer, saved_epoch, run, device):
    global gwriter
    gwriter = writer

    loss = None
    for epoch in range(saved_epoch, args.num_epochs):
        if args.save_freq != 0 and epoch % args.save_freq == 0 and False:
            torch.save({
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch, 'loss': loss,
                'num_workers': args.num_workers, 'device': device,
                'batch_size': args.batch_size,
                'dataset_length': args.dataset_length,
                'num_epochs': args.num_epochs, 'lr': args.lr,
                'lambda_style': args.lambda_style,
                'lambda_content': args.lambda_content,
                'seed': args.seed, 'coco_path': args.coco_path,
                'coco_labels_path': args.coco_labels_path,
                'wiki_path': args.wikiart_path,
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

    progress_bar = tqdm.tqdm(enumerate(dataloader),\
                             total = num_iters, dynamic_ncols = True)
    last_val = 0
    for i, (content_image, style_image) in progress_bar:
        global it
        it = epoch_num * num_iters + i

        content_image = content_image.to(device)
        style_image = style_image.to(device)
        g_batch_size, _, w, h = content_image.shape
        assert_shape(content_image, style_image.shape)

        optimizer.zero_grad()
        style_loss, content_loss, stylized = \
                get_style_transfer_loss(encoder, decoder, \
                                        content_image, style_image,\
                                        args.lambda_content, args.lambda_style)



        if epoch_num == 0 and 10 < i < 30:
            plot_grad(style_loss, content_loss, decoder, optimizer, writer, i)

        full_loss = style_loss + content_loss
        full_loss.backward()


        total_loss += full_loss.item()

        progress_bar.set_postfix({'epoch': f"{epoch_num}",
                                  'loss': f"{total_loss / (i + 1):.2f}"})

        optimizer.step()

        if it % 100 == 0:
            print("validating")
            # == 0 and (torch.cuda.is_available() or epoch_num % 20 == 0):
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

def plot_grad(style_loss, content_loss, decoder, optimizer, writer, it):
    optimizer.zero_grad()
    style_loss.backward(retain_graph = True)
    for tag, param in decoder.named_parameters():
         writer.add_histogram(f"style {tag}", param.grad.data.cpu().numpy(), it)

    optimizer.zero_grad()
    content_loss.backward(retain_graph = True)
    for tag, param in decoder.named_parameters():
         writer.add_histogram(f"content {tag}", param.grad.data.cpu().numpy(), it)

    optimizer.zero_grad()

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


def create_stylized_images(decoder, content_features, style_features):
    shapes = [(g_batch_size, 64, 256, 256), (g_batch_size, 128, 128, 128),\
              (g_batch_size, 256, 64, 64), (g_batch_size, 512, 32, 32)]
    shapes1 = [i.shape for i in content_features]
    shapes2 = [i.shape for i in style_features]
    assert shapes == shapes1 
    assert shapes == shapes2

    stylized_features = adain(content_features[-1], style_features[-1])
    assert_shape(stylized_features, (g_batch_size, 512, 32, 32))

    stylized_images = decoder(stylized_features)
    assert_shape(stylized_images, (g_batch_size, 3, 256, 256))

    return stylized_images, stylized_features


def compute_content_loss(features_of_stylized, stylized_features):
    assert_shape(features_of_stylized, (g_batch_size, 512, 32, 32))
    assert_shape(stylized_features, (g_batch_size, 512, 32, 32))

    content_loss = \
        F.mse_loss(features_of_stylized, stylized_features, reduction = "sum")
    assert_shape(content_loss, ())

    return content_loss


def compute_style_loss(features_of_stylized, style_features):
    shapes = [(g_batch_size, 64, 256, 256), (g_batch_size, 128, 128, 128),\
              (g_batch_size, 256, 64, 64), (g_batch_size, 512, 32, 32)]
    shapes1 = [i.shape for i in features_of_stylized]
    shapes2 = [i.shape for i in style_features]
    assert shapes == shapes1 
    assert shapes == shapes2

    style_loss = 0
    zipped_features = zip(features_of_stylized, style_features)
    scale = [1, 0.8, 0.6, 0.4]
    #scale = [1, 1, 1, 1]
    for i, (feat_of_stylized, style_feat) in enumerate(zipped_features):
        feature_maps = feat_of_stylized.shape[1]
        elems = 2 ** 14 / feature_maps
        shape = (g_batch_size, feature_maps, elems, elems)
        assert_shape(feat_of_stylized, shape)
        assert_shape(style_feat, shape) 

        stdevs1, means1 = calc_feature_stats_vectors(feat_of_stylized)
        stdevs2, means2 = calc_feature_stats_vectors(style_feat)
        assert_shape(stdevs1, (g_batch_size, feature_maps))
        assert_shape(stdevs2, (g_batch_size, feature_maps))
        assert_shape(means1, (g_batch_size, feature_maps))
        assert_shape(means2, (g_batch_size, feature_maps))

        stdev_loss_vector = F.mse_loss(stdevs1, stdevs2, reduction = "mean") ** 0.5 * scale[i]
        mean_loss_vector = F.mse_loss(means1, means2, reduction = "mean") ** 0.5 * scale[i]
        assert_shape(stdev_loss_vector, ())
        assert_shape(mean_loss_vector, ())


        gwriter.add_scalar(f'SLoss/{i}-mlayer', mean_loss_vector.item(), it)
        gwriter.add_scalar(f'SLoss/{i}-slayer', stdev_loss_vector.item(), it)
        style_loss += mean_loss_vector + stdev_loss_vector

    return style_loss


def calc_feature_stats_vectors(features):
    assert len(features.shape) == 4
    batch_size, feature_maps, w, h = features.shape
    assert batch_size == g_batch_size

    features = features.view(batch_size * feature_maps, -1)
    assert features.shape == (g_batch_size * feature_maps, w * h)

    feature_stdevs = features.var(-1, unbiased = False) ** 0.5
    feature_means = features.mean(-1)
    assert_shape(feature_stdevs, (batch_size * feature_maps,))
    assert_shape(feature_means, (batch_size * feature_maps,))

    feature_stdevs = feature_stdevs.reshape(batch_size, feature_maps)
    feature_means = feature_means.reshape(batch_size, feature_maps)

    return feature_stdevs, feature_means


def assert_shape(tensor, expected):
    msg = f"expected {expected} actual {tensor.shape}"
    assert tensor.shape == expected, msg

