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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import random
from torch.utils.tensorboard import SummaryWriter
import argparse


def main():
    global COCO_LABELS_PATH, SEED, BATCH_SIZE, COCO_LABELS_PATH, WIKIART_PATH,\
            NUM_WORKERS, DEVICE, DATASET_LENGTH, LR, LAMBD_STYLE, COCO_PATH, NUM_EPOCHS,\
            LAMBD_CONTENT, LAMBD_STYLE
    has_cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description='Train the model :)')
    parser.add_argument('comment', type = str, help = 'run comment')
    parser.add_argument('--checkpoint', type = str, default=None,
                        help='file to load training from')
    parser.add_argument('--num-workers', type = int,
                        default = os.cpu_count() if has_cuda else 0)
    parser.add_argument('--cpu', type = bool, default = not has_cuda)
    parser.add_argument('--cuda', type = bool, default = has_cuda)
    parser.add_argument('--batch-size', type = int,
                        default = 16 if has_cuda else 2)
    parser.add_argument('--dataset-length', type = int,
                        default = 1000 if has_cuda else 1)
    parser.add_argument('--num-epochs', type = int, default = 10000)
    parser.add_argument('--save-freq', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lambda-style', type = float, default = 10)
    parser.add_argument('--lambda-content', type = float, default = 1)
    parser.add_argument('--seed', type = int, default = 3033157)
    parser.add_argument('--coco-path', type = str,
                        default = "datasets/train2017")
    parser.add_argument('--coco-labels-path', type = str,
                        default = "datasets/annotations/captions_train2017.json")
    parser.add_argument('--wikiart-path', type = str,
                        default = "datasets/wikiart")
    args = parser.parse_args()

    assert args.cpu != args.cuda
    args.device = "cpu" if args.cpu else "cuda"

    if args.checkpoint is None:
        NUM_WORKERS = args.num_workers
        DEVICE = torch.device(args.device)
        SAVE_FREQ = args.save_freq
        BATCH_SIZE = args.batch_size
        DATASET_LENGTH = args.dataset_length
        NUM_EPOCHS = args.num_epochs
        LR = args.lr
        LAMBD_STYLE = args.lambda_style
        LAMBD_CONTENT = args.lambda_content
        SEED = args.seed
        print(f"num_workers: {NUM_WORKERS}, device: {DEVICE}")

        COCO_PATH = args.coco_path
        COCO_LABELS_PATH = args.coco_labels_path
        WIKIART_PATH = args.wikiart_path
        optimizer_state_dict = None
        scheduler_state_dict = None
        decoder_state_dict = None
    else:
        assert len(vars(args)) == 1, \
           "Cannot have multiple args if loading checkpoint."

        checkpoint = torch.load(args.checkpoint)

        optimizer_state_dict = checkpoint['optimizer_state_dict']
        scheduler_state_dict = checkpoint['scheduler_state_dict']
        decoder_state_dict = checkpoint['decoder_state_dict']
        saved_epoch = checkpoint['epoch']
        del checkpoint['epoch']
        print(f"Starting loss: {checkpoint['loss']}")
        del checkpoint['loss']
        NUM_WORKERS = checkpoint['num_workers']
        del checkpoint['num_workers']
        BATCH_SIZE = checkpoint['batch_size']
        del checkpoint['batch_size']
        NUM_EPOCHS = checkpoint['num_epochs']
        del checkpoint['num_epochs']
        LAMBD_STYLE = checkpoint['lambda_style']
        del checkpoint['lambda_style']
        SEED = checkpoint['seed'] + 1
        del checkpoint['seed']
        COCO_LABELS_PATH = checkpoint['coco_labels_path']
        del checkpoint['coco_labels_path']
        DATASET_LENGTH = checkpoint['dataset_length']
        del checkpoint['dataset_length']
        del checkpoint['lr']
        COCO_PATH = checkpoint['coco_path']
        del checkpoint['coco_path']
        WIKIART_PATH = checkpoint['wiki_path']
        del checkpoint['wiki_path']
        LAMBD_CONTENT = checkpoint['lambda_content']
        del checkpoint['lambda_content']
        del checkpoint['device']

        assert len(checkpoint) == 0, checkpoint

        print("Loading checkpoint, ignoring other args")

    encoder = VGG19Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    if decoder_state_dict is not None:
        decoder.load_state_dict(decoder_state_dict)

    encoder.train()
    decoder.train()

    print(encoder)
    print(decoder)
    print("Created models")

    transform = get_transforms()
    dataset = IterableStyleTransferDataset(COCO_PATH, COCO_LABELS_PATH,\
                                   WIKIART_PATH, length = DATASET_LENGTH,
                                   transform = transform, exclude_style = False,
                                   rng_seed = SEED)

    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, \
                            num_workers = NUM_WORKERS)
    print("Created dataloader")

    optimizer = torch.optim.Adam(params = decoder.parameters(), lr = LR)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = ReduceLROnPlateau(optimizer, patience = 100,
                                  verbose = True, threshold = 0)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    saved_epoch = 0


    writer = SummaryWriter(comment = args.comment)
    writer.add_text('batch size', str(BATCH_SIZE), 0)
    writer.add_text('save freq', str(SAVE_FREQ), 0)
    writer.add_text('is cuda ', str(args.cuda), 0)
    writer.add_text('dataset length ', str(DATASET_LENGTH), 0)
    writer.add_text('num workers ', str(NUM_WORKERS), 0)
    writer.add_text('num epochs ', str(NUM_EPOCHS), 0)
    writer.add_text('lambd style', str(LAMBD_STYLE), 0)
    writer.add_text('lambd content', str(LAMBD_CONTENT), 0)
    writer.add_text('encoder', repr(encoder), 0)
    writer.add_text('decoder', repr(decoder), 0)
    writer.add_text('lr', str(LR), 0)

    run = time.time()
    os.makedirs(f"demo/{run}")
    loss = None
    for epoch in range(saved_epoch, NUM_EPOCHS):
        if SAVE_FREQ != 0 and epoch % SAVE_FREQ == 0:
            torch.save({
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch, 'loss': loss,
                'num_workers': NUM_WORKERS, 'device': DEVICE,
                'batch_size': BATCH_SIZE, 'dataset_length': DATASET_LENGTH,
                'num_epochs': NUM_EPOCHS, 'lr': LR,
                'lambda_style': LAMBD_STYLE, 'lambda_content': LAMBD_CONTENT,
                'seed': SEED, 'coco_path': COCO_PATH,
                'coco_labels_path': COCO_LABELS_PATH, 'wiki_path': WIKIART_PATH,
            }, f"demo/{run}/{epoch}.pt")

        loss = train_epoch_style_loss(encoder, decoder, dataloader, \
                                      optimizer, epoch, writer, run)
        scheduler.step(loss)


def train_epoch_style_loss(encoder, decoder, dataloader,\
                           optimizer, epoch_num, writer, run):
    global g_batch_size

    encoder.train()
    decoder.train()

    total_loss = 0
    num_iters = len(dataloader)
    if isinstance(dataloader.dataset, IterableStyleTransferDataset):
        num_iters = ceil(len(dataloader) / BATCH_SIZE)
        print(num_iters)
    print(num_iters)

    progress_bar = tqdm.tqdm(enumerate(dataloader),\
                             total = num_iters, dynamic_ncols = True)
    for i, (content_image, style_image) in progress_bar:

        content_image = content_image.to(DEVICE)
        style_image = style_image.to(DEVICE)
        g_batch_size, _, w, h = content_image.shape
        # assert content_image.shape == style_image.shape

        optimizer.zero_grad()
        style_loss, content_loss, stylized = \
                get_batch_style_transfer_loss(encoder, decoder, \
                                              content_image, style_image)
        full_loss = style_loss + content_loss
        full_loss.backward()

        total_loss += full_loss.item()

        progress_bar.set_postfix({'epoch': f"{epoch_num}",
                                  'loss': f"{total_loss / (i + 1):.2f}"})

        if i == 0:
            s_image = prep_img_for_tb(style_image[0])
            writer.add_image('style', s_image, epoch_num)
            c_image = prep_img_for_tb(content_image[0])
            writer.add_image('content', c_image, epoch_num)
            f_image = prep_img_for_tb(stylized[0])
            writer.add_image('stylized', f_image, epoch_num)

        optimizer.step()

        it = epoch_num * num_iters + i
        writer.add_scalar('SLoss/train_it', style_loss.item(), it)
        writer.add_scalar('CLoss/train_it', content_loss.item(), it)
        writer.add_scalar('Loss/train_it', full_loss.item(), it)



    writer.add_scalar('Loss/train', total_loss, epoch_num)
    return total_loss

def prep_img_for_tb(image):
    copy = image.cpu().detach().clone()
    clamp = torch.clamp(copy, 0, 1)
    # assert clamp.shape == (3, 256, 256)

    return clamp


def get_batch_style_transfer_loss(encoder, decoder, content_image, style_image):
    # assert content_image.shape == (g_batch_size, 3, 256, 256)

    style_features = encoder(style_image)
    content_features = encoder(content_image)

    stylized_images, stylized_features =\
                create_stylized_images(decoder, content_features, style_features)

    features_of_stylized = encoder(stylized_images)

    style_loss = compute_style_loss(features_of_stylized, style_features)
    content_loss = compute_content_loss(features_of_stylized[-1],\
                                        stylized_features)

    return style_loss * LAMBD_STYLE, content_loss * LAMBD_CONTENT,\
                                     stylized_images


def create_stylized_images(decoder, content_features, style_features):
    # assert len(content_features) == 4
    # assert len(style_features) == 4

    """
    shapes = [i.shape for i in content_features]
     assert shapes == [(g_batch_size, 64, 256, 256),
                      (g_batch_size, 128, 128, 128),
                      (g_batch_size, 256, 64, 64),
                      (g_batch_size, 512, 32, 32) ], shapes
    shapes = [i.shape for i in style_features]
    # assert shapes == [(g_batch_size, 64, 256, 256),
                      (g_batch_size, 128, 128, 128),
                      (g_batch_size, 256, 64, 64),
                      (g_batch_size, 512, 32, 32) ], shapes
    """

    stylized_features = adain(content_features[-1], style_features[-1])
    # assert stylized_features.shape == (g_batch_size, 512, 32, 32)

    stylized_images = decoder(stylized_features)
    # assert stylized_images.shape == (g_batch_size, 3, 256, 256)

    return stylized_images, stylized_features


def compute_content_loss(features_of_stylized, stylized_features):
    # assert features_of_stylized.shape == (g_batch_size, 512, 32, 32)
    # assert stylized_features.shape == (g_batch_size, 512, 32, 32)

    batch_size = features_of_stylized.shape[0]

    content_loss = \
        F.mse_loss(features_of_stylized, stylized_features, reduction = "none")
    # assert content_loss.shape == (g_batch_size, 512, 32, 32), content_loss.shape

    content_loss = content_loss.view(batch_size, -1)
    # assert content_loss.shape == (batch_size, 512 * 32 * 32)
    content_loss = content_loss.sum(-1) ** 0.5
    # assert content_loss.shape == (batch_size,)

    return content_loss.mean()


def compute_style_loss(features_of_stylized, style_features):
    style_loss = 0

    """
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
    """

    zipped_features = zip(features_of_stylized, style_features)
    for feat_of_stylized, style_feat in zipped_features:
        stdevs1, means1 = calc_feature_stats_vectors(feat_of_stylized)
        stdevs2, means2 = calc_feature_stats_vectors(style_feat)

        stdev_loss_vector = F.mse_loss(stdevs1, stdevs2, reduction = "none")
        mean_loss_vector = F.mse_loss(means1, means2, reduction = "none")

        stdev_loss_vector = stdev_loss_vector.view(g_batch_size, -1)
        stdev_losses = stdev_loss_vector.sum(-1) ** 0.5
        # assert stdev_losses.shape == (g_batch_size,)
        style_loss += stdev_losses.mean()

        mean_loss_vector = mean_loss_vector.view(g_batch_size, -1)
        mean_losses = mean_loss_vector.sum(-1) ** 0.5
        # assert mean_losses.shape == (g_batch_size,)
        style_loss += mean_losses.mean()

    return style_loss


def calc_feature_stats_vectors(features):
    # assert len(features.shape) == 4
    batch_size, feature_maps, w, h = features.shape
    # assert batch_size == g_batch_size

    features = features.view(batch_size * feature_maps, -1)
    # assert features.shape == (g_batch_size * feature_maps, w * h)

    feature_stdevs = features.var(-1, unbiased = False) ** 0.5
    feature_means = features.mean(-1)
    # assert feature_vars.shape == (batch_size * feature_maps,), feature_vars.shape
    # assert feature_means.shape == (batch_size * feature_maps,), feature_means.shape

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


def train_epoch_reconstruct(encoder, decoder, dataloader, optimizer, epoch_num, writer, run):
    encoder.train()
    decoder.train()

    total_loss = 0
    for i, content_image in tqdm.tqdm(enumerate(dataloader),\
                                      total = len(dataloader),\
                                      dynamic_ncols = True):
        content_image = content_image.to(DEVICE)

        optimizer.zero_grad()
        reconstruction = decoder(encoder(content_image)[-1])

        if i % 300 == 0:
            show_tensor(reconstruction[0].detach().clone(),\
                        epoch_num, run, info = "recon1")
            show_tensor(content_image[0].detach().clone(),\
                        epoch_num, run, info = "orgnl1")
            show_tensor(reconstruction[1].detach().clone(),\
                        epoch_num, run, info = "recon2")
            show_tensor(content_image[1].detach().clone(),\
                        epoch_num, run, info = "orgnl2")

        loss = F.mse_loss(content_image, reconstruction)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar('Loss/train_it', loss.item(), epoch_num)

    writer.add_scalar('Loss/train', total_loss, epoch_num)
    print(f"Epoch {epoch_num}, Loss {total_loss}")
    return total_loss


if __name__ == "__main__":
    main()
