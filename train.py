import os
from math import ceil
import tqdm

import torch
import torch.nn.functional as F

from data import IterableStyleTransferDataset
import adain
from validate import validate_style_loss


def train(adain_model, dataloader, val_dataloader, optimizer, \
          scheduler, args, writer, saved_epoch, run):
    global gwriter
    gwriter = writer

    loss = None
    for epoch_num in range(saved_epoch, args.num_epochs):
        if args.checkpoint_freq != 0 and epoch_num % args.checkpoint_freq == 0:
            torch.save({
                'model_state_dict': adain_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch_num, 'loss': loss,
                'num_workers': args.num_workers,
                'device': args.device,
                'batch_size': args.batch_size,
                'dataset_length': args.dataset_length,
                'num_epochs': args.num_epochs, 'lr': args.lr,
                'lambda_style': args.lambda_style,
                'lambda_content': args.lambda_content,
                'seed': args.seed, 'coco_path': args.coco_path,
                'coco_labels_path': args.coco_labels_path,
                'wiki_path': args.wikiart_path
            }, f"demo/{run}/{epoch_num}.pt")

        loss = train_epoch_style_loss(args, adain_model, dataloader, \
                                      val_dataloader, optimizer,\
                                      epoch_num, writer, run)
        scheduler.step()


def train_epoch_style_loss(args, adain_model, dataloader, val_dataloader,
                           optimizer, epoch_num, writer, run):
    encoder.eval()
    decoder.train()

    total_loss = 0
    num_batches = calc_num_batches(dataloader, args)
    progress_bar = tqdm.tqdm(enumerate(dataloader), total = num_batches, dynamic_ncols = True)
    for i, (content_image, style_image) in progress_bar:
        # move to gpu
        content_images = content_images.to(args.device)
        style_images = style_images.to(args.device)

        # training
        optimizer.zero_grad()
        style_loss, content_loss = get_style_transfer_loss(adain_model, content_images, style_images,\
                                                           args.lambda_content, args.lambda_style)
        loss = style_loss + content_loss
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        # logging
        iteration = epoch_num * num_batches + i
        write_to_tensorboard(iteration, args, adain_model, val_dataloader, writer,
                             style_loss, content_loss)
        progress_bar.set_postfix({'epoch': f"{epoch_num}", 'loss': f"{total_loss / (i + 1):.2f}"})
    writer.add_scalar('Loss/train', total_loss, epoch_num)

    return total_loss


def get_style_transfer_loss(adain_model, content_images, style_images, lambda_content, lambda_style):
    style_features, content_features, stylized_features, stylized_images = adain_model(content_images, style_images)
    features_of_stylized = adain_model.encoder(stylized_images)

    style_loss = lambda_style * calc_style_loss(features_of_stylized, style_features)
    content_loss = lambda_content * calc_content_loss(features_of_stylized[-1], stylized_features)
    return style_loss + content_loss


def calc_content_loss(features_of_stylized, stylized_features):
    return F.mse_loss(features_of_stylized, stylized_features, reduction = "sum")


def calc_style_loss(features_of_stylized, style_features):
    style_loss = 0
    for feat_of_stylized, style_feat in zip(features_of_stylized, style_features):
        stdevs1, means1 = adain.calc_feature_stats(feat_of_stylized)
        stdevs2, means2 = adain.calc_feature_stats(style_feat)

        stdev_loss_vector = F.mse_loss(stdevs1, stdevs2, reduction = "mean") ** 0.5
        mean_loss_vector = F.mse_loss(means1, means2, reduction = "mean") ** 0.5

        gwriter.add_scalar(f'Grads/{i}-mlayer', mean_loss_vector.item(), it)
        gwriter.add_scalar(f'Grads/{i}-slayer', stdev_loss_vector.item(), it)
        style_loss += mean_loss_vector + stdev_loss_vector

    return style_loss


def write_to_tensorboard(iteration, adain_model, val_dataloader, writer, style_loss, content_loss):
    if it % args.save_freq == 0:
        validate_style_loss(adain_model, val_dataloader, it, writer, args.device)
        adain_model.train()

    writer.add_scalar('SLoss/style_it', style_loss.item(), it)
    writer.add_scalar('CLoss/content_it', content_loss.item(), it)
    writer.add_scalar('Loss/total_it', full_loss.item(), it)


def calc_num_batches(dataloader, args):
    if isinstance(dataloader.dataset, IterableStyleTransferDataset):
        return ceil(len(dataloader) / args.batch_size)
    return len(dataloader)


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


def assert_shape(tensor, expected):
    msg = f"expected {expected} actual {tensor.shape}"
    assert tensor.shape == expected, msg

