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
BATCH_SIZE = 32 if cuda else 1
DATASET_LENGTH = 10000
NUM_EPOCHS = 200

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
                                   transform = transform, exclude_style = True)

    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS, shuffle = True)
    print("Created dataloader")

    optimizer = torch.optim.Adam(params = decoder.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = True)

    run = time.time()
    os.makedirs(f"demo/{run}")
    for epoch in range(NUM_EPOCHS):

        if epoch % 7 == 0:
            torch.save(encoder, f"demo/{run}/enc-{epoch}.pt")
            torch.save(decoder, f"demo/{run}/dec-{epoch}.pt")

        loss = train_epoch_reconstruct(encoder, decoder, dataloader, optimizer, epoch, writer, run)
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

def train_epoch_style_loss(encoder, decoder, dataloader, optimizer):
    for i, (content_image, style_image) in enumerate(dataloader):
        content_image, style_image = content_image.to(DEVICE), style_image.to(DEVICE)

        # show_tensor(content_image)
        optimizer.zero_grad()

        loss = get_batch_style_transfer_loss(encoder, decoder, content_image, style_image)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)

        print(f"Loss: {loss.item()}") # TODO: Remove after dev done

        loss.backward()
        optimizer.step()


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



def get_batch_style_transfer_loss(encoder, decoder, content_image, style_image):
    style_features = encoder(style_image)
    content_features = encoder(content_image)

    stylized_image, stylized_features = create_stylized_image(decoder, content_features, style_features)
    if random.random() < 0.1:
        show_tensor(stylized_image.clone().detach())

    features_of_stylized = encoder(stylized_image)

    style_loss = compute_style_loss(features_of_stylized, style_features)
    #content_loss = compute_content_loss(features_of_stylized[-1], stylized_features)
    content_loss = compute_content_loss(content_features[-1], stylized_features)

    return style_loss + content_loss


def create_stylized_image(decoder, content_features, style_features):
    # stylized_content_features = adain(content_features[-1], style_features[-1])
    # stylized_image = decoder(stylized_content_features)
    stylized_image = decoder(content_features[-1])

    return stylized_image, content_features[-1]#stylized_content_features


def compute_content_loss(feature_of_stylized, stylized_features):
    return F.mse_loss(feature_of_stylized, stylized_features, reduction = "mean")


def compute_style_loss(features_of_stylized, style_features):
    return 0
    style_loss = 0

    zipped_features = zip(features_of_stylized, style_features)
    for feat_of_stylized, style_feat in zipped_features:
        stats1 = calc_feature_stats_vector(feat_of_stylized)
        stats2 = calc_feature_stats_vector(style_feat)
        style_loss += F.mse_loss(stats1, stats2, reduction="mean")

    return style_loss




if __name__ == "__main__":
    main()
