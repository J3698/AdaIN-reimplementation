import os
import argparse
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data import StyleTransferDataset, IterableStyleTransferDataset, get_transforms
from train import train
from adain_model import StyleTransferModel


def main():
    args = parse_arguments()
    torch.random.manual_seed(args.seed)

    print("Creating dataloaders")
    dataloader, val_dataloader = create_dataloaders(args)

    print("Creating model")
    adain_model = StyleTransferModel.to(args.device)
    print(adain_model)

    print(f"Workers: {args.num_workers}")
    print(f"Device: {args.device}")

    optimizer = torch.optim.Adam(params = decoder.parameters(), lr = args.lr, weight_decay = 0)
    scheduler = StepLR(optimizer, step_size = args.scheduler_step, gamma = args.scheduler_gamma)

    writer = SummaryWriter(comment = args.comment)
    log_settings(writer, args)

    run = time.time()
    os.makedirs(f"demo/{run}")

    train(adain_model, dataloader, val_dataloader, optimizer,\
          scheduler, args, writer, saved_epoch, run, args.device)


def parse_arguments():
    has_cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description='Train the model :)')
    parser.add_argument('comment', type = str, help = 'run comment')
    parser.add_argument('--num-workers', type = int, default = os.cpu_count() if has_cuda else 0)
    parser.add_argument('--cpu', default = not has_cuda, action='store_true')
    parser.add_argument('--cuda', default = has_cuda, action='store_true')
    parser.add_argument('--crop', default = False, action='store_true')
    parser.add_argument('--batch-size', type = int, default = 16 if has_cuda else 2)
    parser.add_argument('--dataset-length', type = int, default = 1000 if has_cuda else 1)
    parser.add_argument('--num-epochs', type = int, default = 10000)
    parser.add_argument('--log-freq', type = int, default = 100)
    parser.add_argument('--checkpoint-freq', type = int, default = 1)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--lambda-style', type = float, default = 2e6)
    parser.add_argument('--seed', type = int, default = 3033157)
    parser.add_argument('--coco-path', type = str, default = "datasets/train2017")
    parser.add_argument('--coco-labels-path', type = str, default = "datasets/annotations/captions_train2017.json")
    parser.add_argument('--wikiart-path', type = str, default = "datasets/wikiart")
    parser.add_argument('--scheduler-step', type = int, default = 15)
    parser.add_argument('--scheduler-gamma', type = int, default = 0.1)
    args = parser.parse_args()
    assert args.cpu != args.cuda, "Can't train on both cpu and gpu"
    args.device = "cpu" if args.cpu else "cuda"

    return args


def create_dataloaders(args):
    transform = get_transforms(args.crop)
    dataset = IterableStyleTransferDataset(args.coco_path, args.coco_labels_path,\
                                   args.wikiart_path, length = args.dataset_length,
                                   transform = transform, rng_seed = args.seed)
    transform = get_transforms(False)
    val_dataset = StyleTransferDataset(args.coco_path, args.coco_labels_path,\
                                   args.wikiart_path, length = args.dataset_length,
                                   transform = transform, rng_seed = args.seed)
    if isinstance(dataset, IterableStyleTransferDataset):
        dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, num_workers = 1, shuffle = True)

    return dataloader, val_dataloader


def log_settings(writer, args):
    writer.add_text('batch size', str(args.batch_size), 0)
    writer.add_text('save freq', str(args.save_freq), 0)
    writer.add_text('is cuda', str(args.cuda), 0)
    writer.add_text('dataset length', str(args.dataset_length), 0)
    writer.add_text('num workers', str(args.num_workers), 0)
    writer.add_text('num epochs', str(args.num_epochs), 0)
    writer.add_text('lambda style', str(args.lambda_style), 0)
    writer.add_text('model', repr(adain_model), 0)
    writer.add_text('lr', str(args.lr), 0)


if __name__ == "__main__":
    main()
