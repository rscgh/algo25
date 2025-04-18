"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 11/04/25
"""
import argparse
import os
import math
import time
import shutil
import datetime

import torch
import torch.utils.data
import torch.utils.tensorboard
import wandb
import numpy as np

import util
import models

from data.friends import FriendsDataset
from util import warmup_learning_rate, adjust_learning_rate, save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a contrastive video-fmri model on friends dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the trained model")
    parser.add_argument('--downsampled', action='store_true', help='Use downsampled videos')

    # misc
    parser.add_argument('--device', help="device to use", type=str, default='cuda')
    parser.add_argument('--trial', help="random seed / trial id", type=int, default=0)
    parser.add_argument('--amp', action='store_true', help="use automatic mixed precision")
    parser.add_argument('--print_freq', type=int, help='print frequency', default=1)
    parser.add_argument('--minibatch_log_freq', type=int, help='minibatch log frequency', default=50)
    parser.add_argument('--restore', action="store_true", help='restore training')

    # model
    parser.add_argument('--model', help="model to use", type=str, default='vivit-mlp')
    parser.add_argument('--embed_dim', help="embedding dimension", type=int, default=128)
    parser.add_argument('--temperature', help="temperature for clip loss", type=float, default=1.0)

    # optimization
    parser.add_argument('--optimizer', help="optimizer to use", type=str, default='adamw')
    parser.add_argument('--lr', help="learning rate", type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step'], default='step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="7,8,9")
    parser.add_argument('--lr_decay_step', type=int, help='decay rate step (overwrites lr_decay_epochs)', default=10)
    parser.add_argument('--warm', action='store_true', help='warmup learning rate')
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', help="weight decay", type=float, default=1e-5)
    parser.add_argument('--batch_size', help="batch size", type=int, default=256)
    parser.add_argument('--epochs', help="number of epochs", type=int, default=10)
    parser.add_argument('--timesample', help="time downsample factor (reduce memory)", type=int, default=1)

    opts = parser.parse_args()

    if opts.batch_size > 256:
        print("Forcing warm")
        opts.warm = True

    if opts.lr_decay_step is not None:
        opts.lr_decay_epochs = list(range(opts.lr_decay_step, opts.epochs, opts.lr_decay_step))
        print(f"Computed decay epochs based on step ({opts.lr_decay_step}):", opts.lr_decay_epochs)
    else:
        iterations = opts.lr_decay_epochs.split(',')
        opts.lr_decay_epochs = list([])
        for it in iterations:
            opts.lr_decay_epochs.append(int(it))

    if opts.warm:
        opts.warmup_from = 0.01
        opts.warm_epochs = 10
        if opts.lr_decay == 'cosine':
            eta_min = opts.lr * (opts.lr_decay_rate ** 3)
            opts.warmup_to = eta_min + (opts.lr - eta_min) * (
                    1 + math.cos(math.pi * opts.warm_epochs / opts.epochs)) / 2
        else:
            opts.milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
            opts.warmup_to = opts.lr

    return opts


def load_model(opts):
    if opts.model == "vivit-mlp":
        model = models.vivit.VivitMLPContrastive(embed_dim=opts.embed_dim, temperature=opts.temperature).to(opts.device)
        return model.image_processor(), model

    raise ValueError(f"Model not recognized {opts.model}")

def load_optimizer(model, opts):
    if opts.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-8) #, weight_decay=opts.weight_decay)
    elif opts.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, momentum=opts.momentum)

    raise ValueError("Optimizer not recognized")


def train(model, dataloader, optimizer, opts, epoch, writer):
    loss = util.AverageMeter()
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    scaler = torch.amp.GradScaler("cuda", enabled=opts.amp)

    model.train()

    t1 = time.time()
    for idx, (video, fmri) in enumerate(dataloader):
        video, fmri = video.to(opts.device), fmri.to(opts.device)
        data_time.update(time.time() - t1)

        video['pixel_values'] = video['pixel_values'].squeeze(1)
        # print("Video shape:", video['pixel_values'].shape)
        bsz = video['pixel_values'].shape[0]
        warmup_learning_rate(opts, epoch, idx, len(dataloader), optimizer)

        with torch.amp.autocast("cuda", enabled=opts.amp):
            running_loss = model(video, fmri)

        optimizer.zero_grad()
        if opts.amp:
            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            running_loss.backward()
            optimizer.step()

        loss.update(running_loss.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(dataloader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(dataloader)}]:\t"
                  f"DT {data_time.avg:.3f}\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t")

        if (idx + 1) % opts.minibatch_log_freq == 0 or idx == 0:
            writer.add_scalar("train/MB_loss", loss.avg, idx + epoch * len(dataloader))
            writer.add_scalar("MB_lr", optimizer.param_groups[0]['lr'], idx + epoch * len(dataloader))
            writer.add_scalar("MB_BT", batch_time.avg, idx + epoch * len(dataloader))
            writer.add_scalar("MB_DT", data_time.avg, idx + epoch * len(dataloader))
            writer.add_scalar("MB_step", idx + epoch * len(dataloader), idx + epoch * len(dataloader))

    return loss.avg, batch_time.avg, data_time.avg


def main():
    opts = parse_args()
    util.set_seed(opts.trial)

    run_name = (f"{opts.model}_{'downsampled_' if opts.downsampled else ''}{opts.optimizer}_lr{opts.lr}_decay{opts.lr_decay}_"
                f"wd{opts.weight_decay}_bsz{opts.batch_size}_ts{opts.timesample}_"
                f"epochs{opts.epochs}_s{opts.trial}")

    tb_dir = os.path.join(opts.save_dir, "tensorboard", run_name)
    save_dir = os.path.join(opts.save_dir, "models", run_name)
    opts.save_dir = save_dir
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if opts.restore:  # change run name for wandb only (local files will be saved in the same folder)
        run_name = f"{run_name}_restore"

    wandb.init(project="algonauts-challenge-2025", name=run_name, config=opts, sync_tensorboard=True)
    writer = torch.utils.tensorboard.SummaryWriter(tb_dir)

    packages = util.get_packages_versions()
    wandb.config.update({"env": packages})
    print("Packages:")
    for k, v in packages.items():
        print(f"{k}=={v}")

    preprocess, model = load_model(opts)
    optimizer = load_optimizer(model, opts)

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    tot_trainable = sum([np.prod(p.size()) for p in trainable_parameters])
    tot_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total parameters:", tot_parameters, "Trainable parameters:", tot_trainable)

    # Load dataset
    dataset = FriendsDataset(root=opts.data_dir, timesample=opts.timesample, image_transform=preprocess,
                             downsampled=opts.downsampled)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=8,
                                             pin_memory=True, prefetch_factor=2)

    save_file = os.path.join(save_dir, "weights.pth")
    start_epoch = 1
    if opts.restore:
        print("Restoring training....")
        print("Attempting to load", save_file)

        checkpoint = torch.load(save_file, map_location=opts.device, weights_only=False)
        if checkpoint['epoch'] >= opts.epochs:
            print(f"Model already trained for {checkpoint['epoch']} epochs")
            exit(0)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Restored model from epoch {start_epoch}")

        # Copy old file to weights.pth.{epoch}
        shutil.copyfile(save_file, f"{save_file}.{checkpoint['epoch']}")

    print('Config:', opts)
    print('Model:', opts.model, model.__class__.__name__)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)
    print("CUDA available:", torch.cuda.is_available(), f"({torch.cuda.device_count()} devices)")

    start_time = time.time()
    for epoch in range(start_epoch, opts.epochs + 1):
        adjust_learning_rate(opts, optimizer, epoch)

        t1 = time.time()
        loss, batch_time, data_time = train(model, dataloader, optimizer, opts, epoch, writer)
        t2 = time.time()

        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("BT", batch_time, epoch)
        writer.add_scalar("DT", data_time, epoch)
        writer.add_scalar("epoch", epoch, epoch)
        print(f"epoch {epoch}, total time {t2 - start_time:.2f}, epoch time {t2 - t1:.3f} loss {loss:.4f}")

        save_model(model, optimizer, opts, epoch, save_file)

if __name__ == '__main__':
    main()


