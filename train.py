from encoder import VGG19Encoder
from decoder import Decoder
import os
from adain import adain
from math import ceil
import tqdm
from torch.utils.data import DataLoader
from data import IterableStyleTransferDataset, get_transforms
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import random
from torch.utils.tensorboard import SummaryWriter

# Determine constants
cuda = torch.cuda.is_available()
NUM_WORKERS = os.cpu_count() if cuda else 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16 if cuda else 3
DATASET_LENGTH = 1000 if cuda else 1
NUM_EPOCHS = 10000
LR = 1e-3
LAMBD_STYLE = 0.1
LAMBD_CONTENT = 1
SEED = 303

print(f"num_workers: {NUM_WORKERS}, device: {DEVICE}")

COCO_PATH = "datasets/train2017"
COCO_LABELS_PATH = "datasets/annotations/captions_train2017.json"
WIKIART_PATH = "datasets/wikiart"

writer = SummaryWriter()
writer.add_text('batch size', str(BATCH_SIZE), 0)
writer.add_text('is gpu ', str(cuda), 0)
writer.add_text('dataset length ', str(DATASET_LENGTH), 0)
writer.add_text('num workers ', str(NUM_WORKERS), 0)
writer.add_text('num epochs ', str(NUM_EPOCHS), 0)
writer.add_text('lambd style', str(LAMBD_STYLE), 0)
writer.add_text('lambd content', str(LAMBD_CONTENT), 0)

def main():
    encoder = VGG19Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    encoder.train()
    decoder.train()
    print(encoder)
    print(decoder)
    writer.add_text('encoder', repr(encoder), 0)
    writer.add_text('decoder', repr(decoder), 0)
    print("Created models")

    transform = get_transforms()
    dataset = IterableStyleTransferDataset(COCO_PATH, COCO_LABELS_PATH,\
                                   WIKIART_PATH, length = DATASET_LENGTH,
                                   transform = transform, exclude_style = False,
                                   rng_seed = SEED)

    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS)
    print("Created dataloader")

    optimizer = torch.optim.Adam(params = decoder.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 100,
                                                     verbose = True, threshold = 0)

    run = time.time()
    os.makedirs(f"demo/{run}")
    loss = None
    for epoch in range(NUM_EPOCHS):
        torch.save({
            'epoch': epoch,
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'num_workers': NUM_WORKERS, 'device': DEVICE,
            'batch_size': BATCH_SIZE, 'dataset_length': DATASET_LENGTH,
            'num_epochs': NUM_EPOCHS, 'lr': LR,
            'lambda_style': LAMBD_STYLE, 'lambda_content': LAMBD_CONTENT,
            'seed': SEED, 'coco_path': COCO_PATH,
            'coco_labels_path': COCO_LABELS_PATH, 'wiki_path': WIKIART_PATH,
        }, f"demo/{run}/{epoch}.pt")

        loss = train_epoch_style_loss(encoder, decoder, dataloader, optimizer, epoch, writer, run)
        scheduler.step(loss)


def train_epoch_style_loss(encoder, decoder, dataloader, optimizer, epoch_num, writer, run):
    global g_batch_size

    encoder.train()
    decoder.train()

    total_loss = 0
    num_iters = ceil(len(dataloader) / BATCH_SIZE)
    progress_bar = tqdm.tqdm(enumerate(dataloader), total = num_iters, dynamic_ncols = True)
    for i, (content_image, style_image) in progress_bar:

        content_image, style_image = content_image.to(DEVICE), style_image.to(DEVICE)
        g_batch_size, _, w, h = content_image.shape
        # assert content_image.shape == style_image.shape

        optimizer.zero_grad()
        style_loss, content_loss, stylized = get_batch_style_transfer_loss(encoder, decoder, \
                                                                           content_image, style_image)
        full_loss = style_loss + content_loss
        full_loss.backward()

        total_loss += full_loss.item()

        progress_bar.set_postfix({'epoch': f"{epoch_num}", 'loss': f"{total_loss / (i + 1):.2f}"})

        if i % 300 == 0:
            step = (num_iters // 300 + 1) * epoch_num + i // 300
            if epoch_num % 5 == 0:
                writer.add_image('style', prep_img_for_tb(style_image[0]), step)
                writer.add_image('content', prep_img_for_tb(content_image[0]), step)
                writer.add_image('stylized', prep_img_for_tb(stylized[0]), step)

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


def get_batch_style_transfer_loss(encoder, decoder, content_image, style_image,
                                  lambds = LAMBD_STYLE, lambdc = LAMBD_CONTENT):
    # assert content_image.shape == (g_batch_size, 3, 256, 256)

    style_features = encoder(style_image)
    content_features = encoder(content_image)

    stylized_images, stylized_features = create_stylized_images(decoder, content_features, style_features)

    features_of_stylized = encoder(stylized_images)

    style_loss = compute_style_loss(features_of_stylized, style_features)
    content_loss = compute_content_loss(features_of_stylized[-1], stylized_features)

    return style_loss * lambds, content_loss * lambdc, stylized_images


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

    content_loss = F.mse_loss(features_of_stylized, stylized_features, reduction = "none")
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
        vars1, means1 = calc_feature_stats_vectors(feat_of_stylized)
        vars2, means2 = calc_feature_stats_vectors(style_feat)

        var_loss_vector = F.mse_loss(vars1, vars2, reduction = "none")
        mean_loss_vector = F.mse_loss(means1, means2, reduction = "none")

        var_loss_vector = var_loss_vector.view(g_batch_size, -1)
        var_losses = var_loss_vector.sum(-1)
        # assert var_losses.shape == (g_batch_size,)
        style_loss += var_losses.mean()

        mean_loss_vector = mean_loss_vector.view(g_batch_size, -1)
        mean_losses = mean_loss_vector.sum(-1)
        # assert mean_losses.shape == (g_batch_size,)
        style_loss += mean_losses.mean()

    return style_loss


def calc_feature_stats_vectors(features):
    # assert len(features.shape) == 4
    batch_size, feature_maps, w, h = features.shape
    # assert batch_size == g_batch_size

    features = features.view(batch_size * feature_maps, -1)
    # assert features.shape == (g_batch_size * feature_maps, w * h)

    feature_vars = features.var(-1, unbiased = False)
    feature_means = features.mean(-1)
    # assert feature_vars.shape == (batch_size * feature_maps,), feature_vars.shape
    # assert feature_means.shape == (batch_size * feature_maps,), feature_means.shape

    feature_vars = feature_vars.reshape(batch_size, feature_maps)
    feature_means = feature_means.reshape(batch_size, feature_maps)

    return feature_vars, feature_means


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
    for i, content_image in tqdm.tqdm(enumerate(dataloader), total = len(dataloader), dynamic_ncols = True):
        content_image = content_image.to(DEVICE)

        optimizer.zero_grad()
        reconstruction = decoder(encoder(content_image)[-1])

        if i % 300 == 0:
            show_tensor(reconstruction[0].detach().clone(), epoch_num, run, info = "recon1")
            show_tensor(content_image[0].detach().clone(), epoch_num, run, info = "orgnl1")
            show_tensor(reconstruction[1].detach().clone(), epoch_num, run, info = "recon2")
            show_tensor(content_image[1].detach().clone(), epoch_num, run, info = "orgnl2")

        loss = F.mse_loss(content_image, reconstruction)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar('Loss/train_it', loss.item(), epoch_num)

    writer.add_scalar('Loss/train', total_loss, epoch_num)
    print(f"Epoch {epoch_num}, Loss {total_loss}")
    return total_loss


def validate(encoder, decoder, dataloader):
    encoder.eval(); decoder.eval()

    with torch.no_grad():
        for i, (content_image, style_image) in enumerate(dataloader):
            content_image, style_image = content_image.to(DEVICE), style_image.to(DEVICE)

    # TODO: Finish


if __name__ == "__main__":
    main()
