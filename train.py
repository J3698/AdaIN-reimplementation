from encoder import VGG19Encoder
from decoder import Decoder
import os
from adain import adain
import tqdm
from torch.utils.data import DataLoader
from data import StyleTransferDataset, get_transforms
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
BATCH_SIZE = 32 if cuda else 3
DATASET_LENGTH = 10000 if cuda else 1
NUM_EPOCHS = 400
LR = 1e-3

print(f"num_workers: {NUM_WORKERS}, device: {DEVICE}")

COCO_PATH = "datasets/coco/train2017"
COCO_LABELS_PATH = "datasets/coco/annotations/captions_train2017.json"
WIKIART_PATH = "datasets/wikiart"

writer = SummaryWriter()
writer.add_text('batch size', str(BATCH_SIZE), 0)
writer.add_text('is gpu ', str(cuda), 0)
writer.add_text('dataset length ', str(DATASET_LENGTH), 0)
writer.add_text('num workers ', str(NUM_WORKERS), 0)
writer.add_text('num epochs ', str(NUM_EPOCHS), 0)

def main():
    encoder = VGG19Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    print(encoder)
    print(decoder)
    writer.add_text('encoder', repr(encoder), 0)
    writer.add_text('decoder', repr(decoder), 0)
    print("Created models")

    transform = get_transforms()
    dataset = StyleTransferDataset(COCO_PATH, COCO_LABELS_PATH,\
                                   WIKIART_PATH, length = DATASET_LENGTH,
                                   transform = transform, exclude_style = False)

    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS, shuffle = True)
    print("Created dataloader")

    optimizer = torch.optim.Adam(params = decoder.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = True)

    run = time.time()
    os.makedirs(f"demo/{run}")
    for epoch in range(NUM_EPOCHS):
        loss = train_epoch_style_loss(encoder, decoder, dataloader, optimizer, epoch, writer, run)
        scheduler.step(loss)


def train_epoch_reconstruct(encoder, decoder, dataloader, optimizer, epoch_num, writer, run):
    encoder.train()
    decoder.train()

    total_loss = 0
    for i, content_image in tqdm.tqdm(enumerate(dataloader), total = len(dataloader), dynamic_ncols = True):
        content_image = content_image.to(DEVICE)

        optimizer.zero_grad()
        reconstruction = decoder(encoder(content_image)[-1])

        if i == 0:
            show_tensor(reconstruction[0].detach().clone(), epoch_num, run, info = "recon1")
            show_tensor(content_image[0].detach().clone(), epoch_num, run, info = "orgnl1")
            show_tensor(reconstruction[1].detach().clone(), epoch_num, run, info = "recon2")
            show_tensor(content_image[1].detach().clone(), epoch_num, run, info = "orgnl2")

        loss = F.mse_loss(content_image, reconstruction)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    writer.add_scalar('Loss/train', total_loss, epoch_num)
    print(f"Epoch {epoch_num}, Loss {total_loss}")
    return total_loss

def train_epoch_style_loss(encoder, decoder, dataloader, optimizer, epoch_num, writer, run):
    encoder.train()
    decoder.train()

    total_loss = 0
    for i, (content_image, style_image) in tqdm.tqdm(enumerate(dataloader),\
                                             total = len(dataloader), dynamic_ncols = True):

        content_image, style_image = content_image.to(DEVICE), style_image.to(DEVICE)
        optimizer.zero_grad()
        loss, stylized = get_batch_style_transfer_loss(encoder, decoder, content_image, style_image)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()

        if i == 0:
            #show_tensor(style_image[0].detach().clone(), epoch_num, run, info = "style1")
            #show_tensor(content_image[0].detach().clone(), epoch_num, run, info = "content1")
            show_tensor(stylized[0].detach().clone(), epoch_num, run, info = "stylized1")

    writer.add_scalar('Loss/train', total_loss, epoch_num)
    print(f"Epoch {epoch_num}, Loss {total_loss}")
    return total_loss


def get_batch_style_transfer_loss(encoder, decoder, content_image, style_image, lambd = 1e-4):
    style_features = encoder(style_image)
    content_features = encoder(content_image)

    stylized_images, stylized_features = create_stylized_images(decoder, content_features, style_features)

    features_of_stylized = encoder(stylized_images)

    style_loss = torch.tensor(0) #compute_style_loss(features_of_stylized, style_features) * lambd
    content_loss = compute_content_loss(features_of_stylized[-1], stylized_features)
    print(style_loss.item(), content_loss.item())

    return style_loss + content_loss, stylized_images


def create_stylized_images(decoder, content_features, style_features):
    stylized_features = adain(content_features[-1], style_features[-1])
    stylized_images = decoder(stylized_features)

    return stylized_images, stylized_features


def compute_content_loss(features_of_stylized, stylized_features):
    return F.mse_loss(features_of_stylized, stylized_features, reduction = "mean")


def compute_style_loss(features_of_stylized, style_features):
    style_loss = 0

    zipped_features = zip(features_of_stylized, style_features)
    for feat_of_stylized, style_feat in zipped_features:
        stats1 = calc_feature_stats_vector(feat_of_stylized)
        stats2 = calc_feature_stats_vector(style_feat)
        style_loss = style_loss + F.mse_loss(stats1, stats2, reduction = "mean")

    return style_loss


def calc_feature_stats_vector(features):
    assert len(features.shape) == 4
    batch_size, feature_maps, w, h = features.shape

    features = features.view(batch_size * feature_maps, -1)

    feature_vars = features.var(-1)
    feature_means = features.mean(-1)
    assert feature_vars.shape == (batch_size * feature_maps,), feature_vars.shape
    assert feature_means.shape == (batch_size * feature_maps,), feature_means.shape

    feature_vars = feature_vars.reshape(batch_size, feature_maps)
    feature_means = feature_means.reshape(batch_size, feature_maps)
    stats_vectors = torch.hstack((feature_vars, feature_means))
    assert stats_vectors.shape == (batch_size, 2 * feature_maps)

    return stats_vectors


def show_tensor(tensor, num, run, info = ""):
    if info != "":
        info = "-" + info

    image = tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    image[image > 1] = 1
    image[image < 0] = 0
    plt.imshow(image)
    plt.savefig(f"demo/{run}/{num}{info}.png")


def validate(encoder, decoder, dataloader):
    encoder.eval(); decoder.eval()

    with torch.no_grad():
        for i, (content_image, style_image) in enumerate(dataloader):
            content_image, style_image = content_image.to(DEVICE), style_image.to(DEVICE)

    # TODO: Finish


if __name__ == "__main__":
    main()
