from encoder import VGG19Encoder
from decoder import Decoder
from adain import adain
from torch.utils.data import DataLoader
from data import StyleTransferDataset, get_transforms
import torch

cuda = torch.cuda.is_available()
NUM_WORKERS = os.cpu_count() if cuda else 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"num_workers: {NUM_WORKERS}, device: {DEVICE}")

def main():
    encoder = VGG19Encoder()
    encoder.freeze()
    decoder = Decoder()
    print("Created models")

    transform = get_transforms()
    dataset = StyleTransferDataset("/Users/jacoblee628/Documents/Github/AdaIN-reimplementation/coco/train2017",  "/Users/jacoblee628/Documents/Github/AdaIN-reimplementation/coco/annotations/captions_train2017.json", "/Users/jacoblee628/Documents/Github/AdaIN-reimplementation/coco/wikiart", transform)

    # dataset = StyleTransferDataset("/home/anti/coco/train2017",  "/home/anti/coco/annotations/captions_train2017.json", "/home/anti/coco/wikiart", transform)
    dataloader = DataLoader(dataset, batch_size = 1, num_workers = 0)
    print("Created dataloader")

    optimizer = torch.optim.Adam(params = decoder.parameters())

    for i, (content_image, style_image) in enumerate(dataloader):
        content_image, style_image = content_image.to(DEVICE), style_image.to(DEVICE)
        #optimizer.zero_grad()

        loss = get_batch_style_transfer_loss(encoder, decoder, content_image, style_image)

        loss.backward()
        #optimizer.step()


def get_batch_style_transfer_loss(encoder, decoder, style_image, content_image):
    style_features = encoder(style_image)
    content_features = encoder(content_image)

    stylized_image, stylized_features = create_stylized_image(decoder, content_features, style_features)

    features_of_stylized = encoder(stylized_image)

    style_loss = compute_style_loss(features_of_stylized, style_features)
    content_loss = compute_content_loss(features_of_stylized, stylized_features)

    return style_loss + content_loss


def create_stylized_image(decoder, content_features, style_features):
    stylized_content_features = adain(content_features[-1], style_features[-1])
    stylized_image = decoder(stylized_content_features)

    return stylized_image, stylized_content_features




if __name__ == "__main__":
    main()
