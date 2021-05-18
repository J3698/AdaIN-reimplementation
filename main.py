from encoder import VGG19Encoder
from train import train
from decoder import Decoder
import os
from adain import adain
from math import ceil
import tqdm
from torch.utils.data import DataLoader
from data import StyleTransferDataset, IterableStyleTransferDataset, get_transforms
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn.functional as F
import time
import random
from torch.utils.tensorboard import SummaryWriter
import argparse


def main():
    has_cuda = torch.cuda.is_available()

    parser = argparse.ArgumentParser(description='Train the model :)')
    parser.add_argument('comment', type = str, help = 'run comment')
    parser.add_argument('--checkpoint', type = str, default=None,
                        help='file to load training from')
    parser.add_argument('--num-workers', type = int,
                        default = os.cpu_count() if has_cuda else 0)
    parser.add_argument('--cpu', default = not has_cuda, action='store_true')
    parser.add_argument('--cuda', default = has_cuda, action='store_true')
    parser.add_argument('--crop', default = True, action='store_true')
    parser.add_argument('--batch-size', type = int,
                        default = 16 if has_cuda else 2)
    parser.add_argument('--dataset-length', type = int,
                        default = 1000 if has_cuda else 1)
    parser.add_argument('--num-epochs', type = int, default = 10000)
    parser.add_argument('--save-freq', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--lambda-style', type = float, default = 30)
    parser.add_argument('--lambda-content', type = float, default = 1)
    parser.add_argument('--seed', type = int, default = 3033157)
    parser.add_argument('--coco-path', type = str,
                        default = "datasets/train2017")
    parser.add_argument('--coco-labels-path', type = str,
                        default = "datasets/annotations/captions_train2017.json")
    parser.add_argument('--wikiart-path', type = str,
                        default = "datasets/wikiart")
    parser.add_argument('--scheduler-step', type = int,
                        default = 200)
    args = parser.parse_args()

    assert args.cpu != args.cuda
    args.device = "cpu" if args.cpu else "cuda"

    saved_epoch = 0
    optimizer_state_dict = None
    scheduler_state_dict = None
    decoder_state_dict = None

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        del checkpoint['optimizer_state_dict']
        scheduler_state_dict = checkpoint['scheduler_state_dict']
        del checkpoint['scheduler_state_dict']
        decoder_state_dict = checkpoint['decoder_state_dict']
        del checkpoint['decoder_state_dict']
        saved_epoch = checkpoint['epoch']
        del checkpoint['epoch']
        print(f"Starting loss: {checkpoint['loss']}")
        del checkpoint['loss']
        args.num_workers = checkpoint['num_workers']
        del checkpoint['num_workers']
        args.batch_size = checkpoint['batch_size']
        del checkpoint['batch_size']
        args.num_epochs = checkpoint['num_epochs']
        del checkpoint['num_epochs']
        args.lambda_style = checkpoint['lambda_style']
        del checkpoint['lambda_style']
        args.seed = checkpoint['seed'] + 1
        del checkpoint['seed']
        coco_labels_path = checkpoint['coco_labels_path']
        del checkpoint['coco_labels_path']
        dataset_length = checkpoint['dataset_length']
        del checkpoint['dataset_length']
        args.lr = checkpoint['lr']
        del checkpoint['lr']
        args.coco_path = checkpoint['coco_path']
        del checkpoint['coco_path']
        args.wikiart_path = checkpoint['wiki_path']
        del checkpoint['wiki_path']
        args.lambda_content = checkpoint['lambda_content']
        del checkpoint['lambda_content']
        device = torch.device(checkpoint['device'])
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        del checkpoint['device']
        if 'crop' in checkpoint:
            crop = checkpoint['crop']
            del checkpoint['crop']
        else:
            crop = True
        if 'save_freq' in checkpoint:
            save_freq = checkpoint['save_freq']
            del checkpoint['save_freq']
        else:
            save_freq = 1

        assert len(checkpoint) == 0, checkpoint
        print("Loading checkpoint, ignoring other args")

    device = torch.device(args.device)
    torch.random.manual_seed(args.seed)
    print(f"num_workers: {args.num_workers}, device: {args.device}")

    encoder = VGG19Encoder().to(device)
    decoder = Decoder().to(device)
    if decoder_state_dict is not None:
        decoder.load_state_dict(decoder_state_dict)

    encoder.train()
    decoder.train()

    print(encoder)
    print(decoder)
    print("Created models")

    transform = get_transforms(args.crop)
    dataset = StyleTransferDataset(args.coco_path, args.coco_labels_path,\
                                   args.wikiart_path, length = args.dataset_length,
                                   transform = transform, rng_seed = args.seed)
    transform = get_transforms(False)
    val_dataset = StyleTransferDataset(args.coco_path, args.coco_labels_path,\
                                   args.wikiart_path, length = args.dataset_length,
                                   transform = transform, rng_seed = args.seed)

    dataloader = DataLoader(dataset, batch_size = args.batch_size, \
                            num_workers = args.num_workers, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, \
                                num_workers = 1, shuffle = True)
    print("Created dataloader")

    optimizer = torch.optim.Adam(params = decoder.parameters(),
                                 lr = args.lr, weight_decay = 0)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = StepLR(optimizer, step_size = args.scheduler_step, gamma = 0.1)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    writer = SummaryWriter(comment = args.comment)
    writer.add_text('batch size', str(args.batch_size), 0)
    writer.add_text('save freq', str(args.save_freq), 0)
    writer.add_text('is cuda ', str(args.cuda), 0)
    writer.add_text('dataset length ', str(args.dataset_length), 0)
    writer.add_text('num workers ', str(args.num_workers), 0)
    writer.add_text('num epochs ', str(args.num_epochs), 0)
    writer.add_text('lambda style', str(args.lambda_style), 0)
    writer.add_text('lambda content', str(args.lambda_content), 0)
    writer.add_text('encoder', repr(encoder), 0)
    writer.add_text('decoder', repr(decoder), 0)
    writer.add_text('lr', str(args.lr), 0)

    run = time.time()
    os.makedirs(f"demo/{run}")

    train(encoder, decoder, dataloader, val_dataloader, optimizer, scheduler, args, writer, saved_epoch, run, device)

if __name__ == "__main__":
    main()
