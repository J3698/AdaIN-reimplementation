import torch
import warnings
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os
from torchvision.transforms import ToTensor, RandomCrop, Resize, Compose
from torchvision.datasets import ImageFolder, CocoCaptions
from random import randrange, seed


def main():
    transforms = get_transforms()
    dataset = StyleTransferDataset("/home/anti/coco/train2017",  "/home/anti/coco/annotations/captions_train2017.json", "/home/anti/coco/wikiart", transform)
    print(f"Dataset length: {len(dataset)}, Wiki length: len(dataset.wiki), COCO length: len(dataset.coco)")

    content, style = dataset[0]
    print(f"1st content img: {type(content)}, {content.shape}")
    print(f"1st style img: {type(style)}, {style.shape}")


def get_transforms():
    return Compose([Resize(512), RandomCrop(256), ToTensor()])
    # return Compose([Resize((256, 256)), ToTensor()])


class StyleTransferDataset(Dataset):
    def __init__(self, coco_path, coco_annotations, \
                 wiki_path, length = 100000, transform = None, rng_seed = 1, exclude_style = False):

        self.wiki = ImageFolder(wiki_path, transform = transform)
        self.coco = CocoCaptions(coco_path, coco_annotations, transform = transform)
        self.length = length
        self.exclude_style = exclude_style

        seed(rng_seed)
        print(length)
        self.indices = [self.random_pair_of_indices() for i in range(length)]


    def random_pair_of_indices(self):
        return randrange(len(self.coco)), randrange(len(self.wiki))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        coco_idx, wiki_idx = self.indices[idx]

        content_image = self.coco[coco_idx][0]
        if not self.exclude_style:
            style_image = self.wiki[wiki_idx][0]
        else:
            style_image = None

        self._check_data_to_return(style_image, content_image)

        if self.exclude_style:
            return content_image
        else:
            return content_image, style_image


    def _check_data_to_return(self, style_image, content_image):
        if not (style_image is None or isinstance(style_image, torch.Tensor)) or \
           not (content_image is None or isinstance(content_image, torch.Tensor)):
            warnings.warn("Given transform does not convert images to tensors;"
                          "default collate may fail.", UserWarning)



if __name__ == "__main__":
    main()
