import torch
import warnings
import torchvision
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.datasets.folder import default_loader
import os
from torchvision.transforms import ToTensor, RandomCrop, Resize, Compose
from torchvision.datasets import ImageFolder, CocoCaptions
from random import randrange, seed
import random
from math import ceil

# https://stackoverflow.com/a/47958486/4142985
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_transforms():
    return Compose([Resize((256, 256)), ToTensor()])
    # return Compose([Resize((256, 256)), ToTensor()])


class IterableStyleTransferDataset(IterableDataset):
    def __init__(self, coco_path, coco_annotations, \
                 wiki_path, length = 100000, transform = None, rng_seed = 1, exclude_style = False):

        self.wiki = ImageFolder(wiki_path, transform = transform)
        self.coco = CocoCaptions(coco_path, coco_annotations, transform = transform)
        self.length = length
        self.exclude_style = exclude_style

        self.seed = rng_seed
        self.random = random.Random(rng_seed)
        print(length)
        self.indices = [self.random_pair_of_indices() for i in range(length)]


    def random_pair_of_indices(self):
        return self.random.randrange(len(self.coco)),\
               self.random.randrange(len(self.wiki))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.random.seed(round(1000 * random.random()))

            length = ceil(self.length // worker_info.num_workers)
            if worker_info.id == worker_info.num_workers - 1:
                length = self.length - length * (worker_info.num_workers - 1)
                assert length > 0
            self.ilength = length
        else:
            self.ilength = length

        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.ilength <= 0:
            raise StopIteration

        self.ilength -= 1

        coco_idx, wiki_idx = self.random_pair_of_indices()

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

class StyleTransferDataset(Dataset):
    def __init__(self, coco_path, coco_annotations, \
                 wiki_path, length = 100000, transform = None, rng_seed = 1, exclude_style = False):

        self.wiki = ImageFolder(wiki_path, transform = transform)
        self.coco = CocoCaptions(coco_path, coco_annotations, transform = transform)
        self.length = length
        self.exclude_style = exclude_style

        seed(rng_seed)
        random.random()
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
