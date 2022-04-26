import os
from math import ceil
import tqdm

import torch
import torch.nn.functional as F

from data import IterableStyleTransferDataset
import adain
from validate import validate_style_loss


def train(adain_model, dataloader, val_dataloader, optimizer, \
          scheduler, args, writer, run):
    global gwriter
    gwriter = writer

    loss = None
    for epoch_num in range(args.num_epochs):
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
        writer.add_scalar('LR/train', optimizer.param_groups[0]['lr'], \
                          epoch_num)
        scheduler.step()


def train_epoch_style_loss(args, adain_model, dataloader, val_dataloader,
                           optimizer, epoch_num, writer, run):
    adain_model.train()

    total_loss = 0
    num_batches = len(dataloader)
    progress_bar = tqdm.tqdm(enumerate(dataloader), total = num_batches,\
                             dynamic_ncols = True)
    for i, (content_images, style_images) in progress_bar:
        # move to gpu
        content_images = content_images.to(args.device)
        style_images = style_images.to(args.device)

        # training
        optimizer.zero_grad()
        style_loss, content_loss = get_style_transfer_loss(adain_model,\
                                       content_images, style_images,\
                                       args.lambda_style, args.lambda_content,
                                       args.swag)
        loss = style_loss + content_loss
        loss.backward()
        optimizer.step()

        # logging
        total_loss += loss.item()
        iteration = epoch_num * num_batches + i
        write_to_tensorboard(iteration, args, adain_model, val_dataloader, \
                             writer, style_loss, content_loss)
        progress_bar.set_postfix({'epoch': f"{epoch_num}", \
                                  'loss': f"{total_loss / (i + 1):.5f}"})
    writer.add_scalar('Loss/train', total_loss, epoch_num)

    return total_loss


def get_style_transfer_loss(adain_model, content_images, style_images, \
                            lambda_style, lambda_content, swag):
    style_features, content_features, stylized_features, stylized_images = \
        adain_model(content_images, style_images)
    features_of_stylized = adain_model.encoder(stylized_images)

    style_loss = lambda_style * calc_style_loss(features_of_stylized, \
                                                style_features, swag)
    content_loss = lambda_content * calc_content_loss(features_of_stylized[-1],\
                                                      stylized_features, swag)
    return style_loss, content_loss


def softmax_feature(feature):
    assert len(feature.shape) == 4
    new_shape = feature.shape[:2] + (-1,)
    feature_softmax = F.softmax(feature.reshape(new_shape), dim = -1)
    return feature_softmax.reshape(feature.shape)


def calc_content_loss(features_of_stylized, stylized_features, swag = False):
    f = softmax_feature if swag else (lambda x: x)
    return F.mse_loss(f(features_of_stylized), f(stylized_features))


def calc_style_loss(features_of_stylized, style_features, swag = False):
    batch_size = features_of_stylized[0].shape[0]
    style_loss = 0
    swag = False
    f = softmax_feature if swag else (lambda x: x)
    for feat_of_stylized, style_feat in zip(features_of_stylized, \
                                            style_features):
        feat_of_stylzied = f(feat_of_stylized)
        style_feat = f(style_feat)

        stdevs1, means1 = adain.calc_feature_stats(feat_of_stylized)
        stdevs2, means2 = adain.calc_feature_stats(style_feat)

        stdev_loss_vector = F.mse_loss(stdevs1.view(batch_size, -1), \
                                       stdevs2.view(batch_size, -1)) ** 0.5
        mean_loss_vector = F.mse_loss(means1.view(batch_size, -1), \
                                      means2.view(batch_size, -1)) ** 0.5

        style_loss += mean_loss_vector + stdev_loss_vector

    return style_loss


def write_to_tensorboard(iteration, args, adain_model, val_dataloader, \
                         writer, style_loss, content_loss):
    if iteration % args.log_freq == 0:
        validate_style_loss(adain_model, val_dataloader, iteration, \
                            writer, args.device)
        adain_model.train()

    writer.add_scalar('SLoss/style_it', style_loss.item(), iteration)
    writer.add_scalar('CLoss/content_it', content_loss.item(), iteration)
    full_loss = style_loss.item() + content_loss.item()
    writer.add_scalar('Loss/total_it', full_loss, iteration)




def plot_grad(style_loss, content_loss, decoder, optimizer, writer, it):
    optimizer.zero_grad()
    style_loss.backward(retain_graph = True)
    for tag, param in decoder.named_parameters():
         writer.add_histogram(f"style {tag}", param.grad.data.cpu().numpy(), it)

    optimizer.zero_grad()
    content_loss.backward(retain_graph = True)
    for tag, param in decoder.named_parameters():
         param = param.grad.data.cpu().numpy()
         writer.add_histogram(f"content {tag}", param, it)

    optimizer.zero_grad()


def assert_shape(tensor, expected):
    msg = f"expected {expected} actual {tensor.shape}"
    assert tensor.shape == expected, msg

