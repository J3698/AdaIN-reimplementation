import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os
from torchvision.transforms import ToTensor, RandomCrop, Resize, Compose
from torchvision.datasets import ImageFolder, CocoCaptions



def main():
    transform = Compose([Resize(512), RandomCrop(256), ToTensor()])
    dataset = StyleTransferDataset("/home/anti/coco/train2017",  "/home/anti/coco/annotations/captions_train2017.json", "/home/anti/coco/wikiart", transform)
    print(f"Dataset length: {len(dataset)}, Wiki length: len(dataset.wiki), COCO length: len(dataset.coco)")

    content, style = dataset[0]
    print(f"1st content img: {type(content)}, {content.shape}")
    print(f"1st style img: {type(style)}, {style.shape}")



class StyleTransferDataset(Dataset):
    def __init__(self, coco_path, coco_annotations, wiki_path, transform = None):
        self.wiki = ImageFolder(wiki_path, transform = transform)
        self.coco = CocoCaptions(coco_path, coco_annotations, transform = transform)


    def __len__(self):
        return len(self.coco) * len(self.wiki)


    def __getitem__(self, idx):
        content = self.coco[idx % len(self.coco)][0]
        style = self.wiki[idx // len(self.coco)][0]

        return content, style



if __name__ == "__main__":
    main()
