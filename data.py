import torch
import torchvision
from torch.utils.data import IterableDataset
from torchvision.datasets.folder import default_loader
import os
from torchvision.transforms import ToTensor, RandomCrop, Resize, Compose
from torchvision.datasets import ImageFolder, CocoCaptions
from random import randrange



def main():
    transforms = get_transforms()
    dataset = StyleTransferDataset("/home/anti/coco/train2017",  "/home/anti/coco/annotations/captions_train2017.json", "/home/anti/coco/wikiart", transform)
    print(f"Dataset length: {len(dataset)}, Wiki length: len(dataset.wiki), COCO length: len(dataset.coco)")

    content, style = dataset[0]
    print(f"1st content img: {type(content)}, {content.shape}")
    print(f"1st style img: {type(style)}, {style.shape}")


def get_transforms():
    return Compose([Resize(512), RandomCrop(256), ToTensor()])

class StyleTransferDataset(IterableDataset):
    def __init__(self, coco_path, coco_annotations, wiki_path, length = 10000, transform = None):
        self.wiki = ImageFolder(wiki_path, transform = transform)
        self.coco = CocoCaptions(coco_path, coco_annotations, transform = transform)


    def __iter__(self):
        self.count = 0
        return self


    def __next__(self):
        if count > self.length:
            raise StopIteration
        self.count += 1

        coco_idx = random.randrange(len(self.coco))
        wiki_idx = random.randrange(len(self.wiki))

        content_image = self.coco[coco_idx][0]
        style_image = self.wiki[wiki_idx][0]

        return content_image, style_image

    def __getitem__(self, idx):
        content_image = self.coco[idx % len(self.coco)][0]
        style_image = self.wiki[idx // len(self.coco)][0]

        return content_image, style_image



if __name__ == "__main__":
    main()
