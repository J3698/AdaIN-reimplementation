from encoder import VGG19Encoder
from decoder import Decoder
from adain import adain
from torch.utils.data import DataLoader
from data import StyleTransferDataset, get_transforms
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import random

# Determine constants
cuda = torch.cuda.is_available()
NUM_WORKERS = os.cpu_count() if cuda else 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"num_workers: {NUM_WORKERS}, device: {DEVICE}")

COCO_PATH = "datasets/coco/train2017"
COCO_LABELS_PATH = "datasets/coco/annotations/captions_train2017.json"
WIKIART_PATH = "datasets/wikiart"

torch.autograd.set_detect_anomaly(True)

def main():
    # init objects
    encoder = VGG19Encoder()
    decoder = Decoder()
    print(decoder)
    print("Created models")

    transform = get_transforms()
    dataset = StyleTransferDataset(COCO_PATH, COCO_LABELS_PATH,\
                                   WIKIART_PATH, length = 1, transform = transform)

    dataloader = DataLoader(dataset, batch_size = 1, num_workers = 0)
    print("Created dataloader")

    num_epochs = 100
    optimizer = torch.optim.Adam(params = decoder.parameters(), lr = 1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        train_epoch_reconstruct(encoder, decoder, dataloader, optimizer, epoch)
        scheduler.step()


def train_epoch_reconstruct(encoder, decoder, dataloader, optimizer, epoch_num):
    for i, (content_image, style_image) in enumerate(dataloader):
        content_image, _ = content_image.to(DEVICE), style_image.to(DEVICE)

        optimizer.zero_grad()
        reconstruction = decoder(encoder(content_image)[-1])

        if epoch_num == 0:
            show_tensor(content_image.detach().clone())

        if epoch_num % 7 == 0:
            show_tensor(reconstruction.detach().clone())

        loss = F.mse_loss(content_image, reconstruction)
        loss.backward()
        optimizer.step()

        print(f"(Debug) {epoch_num} Loss:{loss.item()}") # TODO: Remove after dev done


def train_epoch_style_loss(encoder, decoder, dataloader, optimizer):
    for i, (content_image, style_image) in enumerate(dataloader):
        content_image, style_image = content_image.to(DEVICE), style_image.to(DEVICE)

        # show_tensor(content_image)
        optimizer.zero_grad()

        loss = get_batch_style_transfer_loss(encoder, decoder, content_image, style_image)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5)

        print(f"(Debug) Loss:{loss.item()}") # TODO: Remove after dev done

        loss.backward()
        optimizer.step()


def show_tensor(tensor):
    image = tensor.squeeze().permute(1, 2, 0).numpy()
    plt.imshow(image)
    plt.savefig(f"demo/{time.time()}.png")


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
