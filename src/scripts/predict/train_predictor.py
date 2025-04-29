"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 20/04/25
"""
import argparse
import os
import math
import time
import shutil
import datetime

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.tensorboard
import wandb
import numpy as np

from scipy.stats import pearsonr

import util
import models

from data.friends import FriendsFeatureDataset
from util import warmup_learning_rate, adjust_learning_rate, save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train movie features->fmri predictors on friends dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory containing the features")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory to save the logs")

    # misc
    parser.add_argument('--device', help="device to use", type=str, default='cuda')
    parser.add_argument('--trial', help="random seed / trial id", type=int, default=0)
    parser.add_argument('--amp', action='store_true', help="use automatic mixed precision")
    parser.add_argument('--print_freq', type=int, help='print frequency', default=50)
    parser.add_argument('--minibatch_log_freq', type=int, help='minibatch log frequency', default=50)

    # model
    parser.add_argument('--model', help="model to use", type=str, default='mlp-small')
    parser.add_argument('--dropout', type=float, help="dropout", default=0.0)
    parser.add_argument('--stimulus_window', type=int, help="width of stimulus window (num. of chunks)", default=1)

    # optimization
    parser.add_argument('--optimizer', help="optimizer to use", type=str, default='adam')
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
    if opts.model == "mlp-small":
        return models.predictors.MLPSmall(
            768 * opts.stimulus_window,
            1000,
            opts.dropout
        ).to(opts.device)

    raise ValueError(f"Model not recognized {opts.model}")

def load_optimizer(model, opts):
    if opts.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-8) #, weight_decay=opts.weight_decay)
    elif opts.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, momentum=opts.momentum)

    raise ValueError("Optimizer not recognized")


def torch_pearsonr(output, target):
    x = output
    y = target

    vx = x - torch.mean(x, dim=-1, keepdim=True)
    vy = y - torch.mean(y, dim=-1, keepdim=True)

    return torch.mean(torch.sum(vx * vy, dim=-1) / (torch.sqrt(torch.sum(vx ** 2, dim=-1)) * torch.sqrt(torch.sum(vy ** 2, dim=-1))))


def train(model, dataloader, optimizer, opts, epoch, writer):
    loss = util.AverageMeter()
    # corr = util.AverageMeter()
    batch_time = util.AverageMeter()
    data_time = util.AverageMeter()
    scaler = torch.amp.GradScaler("cuda", enabled=opts.amp)

    all_outputs = []
    all_labels = []

    model.train()

    t1 = time.time()
    for idx, (features, fmri) in enumerate(dataloader):
        features, fmri = features.to(opts.device), fmri.to(opts.device)
        data_time.update(time.time() - t1)

        bsz = features.shape[0]
        warmup_learning_rate(opts, epoch, idx, len(dataloader), optimizer)

        with torch.amp.autocast("cuda", enabled=opts.amp):
            running_loss, outputs = model(features, fmri)

        optimizer.zero_grad()
        if opts.amp:
            scaler.scale(running_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            running_loss.backward()
            optimizer.step()

        loss.update(running_loss.item(), bsz)
        # corr.update(running_r.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(dataloader) - idx)
        all_outputs.append(outputs.detach())
        all_labels.append(fmri)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(dataloader)}]:\t"
                  f"DT {data_time.avg:.3f}\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}")

        # if (idx + 1) % opts.minibatch_log_freq == 0 or idx == 0:
        #     writer.add_scalar("train/MB_loss", loss.avg, idx + epoch * len(dataloader))
        #     writer.add_scalar("MB_lr", optimizer.param_groups[0]['lr'], idx + epoch * len(dataloader))
        #     writer.add_scalar("MB_BT", batch_time.avg, idx + epoch * len(dataloader))
        #     writer.add_scalar("MB_DT", data_time.avg, idx + epoch * len(dataloader))
        #     writer.add_scalar("MB_step", idx + epoch * len(dataloader), idx + epoch * len(dataloader))

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    r = torch_pearsonr(all_outputs, all_labels).item()
    print("r:", r)

    return loss.avg, r, batch_time.avg, data_time.avg


@torch.inference_mode()
def test(model, dataloader, optimizer, opts, epoch, writer):
    model.eval()

    all_outputs = []
    all_labels = []

    for idx, (features, fmri) in enumerate(dataloader):
        features, fmri = features.to(opts.device), fmri.to(opts.device)

        with torch.amp.autocast("cuda", enabled=opts.amp):
            _, outputs = model(features, fmri)

        all_outputs.append(outputs.cpu())
        all_labels.append(fmri.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    mae = F.l1_loss(all_outputs, all_labels)

    # compute average correlation
    r = torch_pearsonr(all_outputs, all_labels).item()
    print("test r:", r)

    return mae, r


def run_training(opts, subject, writer):
    model = load_model(opts)
    optimizer = load_optimizer(model, opts)

    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    tot_trainable = sum([np.prod(p.size()) for p in trainable_parameters])
    tot_parameters = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total parameters:", tot_parameters, "Trainable parameters:", tot_trainable)

    # Load dataset
    train_dataset = FriendsFeatureDataset(root=opts.data_dir, features_root=opts.features_dir,
                                          subjects=[subject], seasons=[1,2,3,4,5],
                                          stimulus_window=opts.stimulus_window)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size,
                                               shuffle=True, num_workers=8, pin_memory=True)

    test_dataset = FriendsFeatureDataset(root=opts.data_dir, features_root=opts.features_dir,
                                         subjects=[subject], seasons=[6],
                                         stimulus_window=opts.stimulus_window)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size,
                                              shuffle=False, num_workers=8, pin_memory=True)

    save_file = os.path.join(opts.save_dir, f"predictor_sub-{subject}.pth")
    start_epoch = 1

    print('Config:', opts)
    print('Model:', opts.model, model.__class__.__name__)
    print('Optimizer:', optimizer)
    print('Scheduler:', opts.lr_decay)
    print("CUDA available:", torch.cuda.is_available(), f"({torch.cuda.device_count()} devices)")

    start_time = time.time()
    for epoch in range(start_epoch, opts.epochs + 1):
        adjust_learning_rate(opts, optimizer, epoch)

        t1 = time.time()
        loss, corr, batch_time, data_time = train(model, train_loader, optimizer, opts, epoch, writer)
        t2 = time.time()

        test_mae, test_r = test(model, test_loader, optimizer, opts, epoch, writer)

        writer.add_scalar(f"sub-{subject}/train/loss", loss, epoch)
        writer.add_scalar(f"sub-{subject}/train/r", corr, epoch)
        writer.add_scalar(f"sub-{subject}/test/MAE", test_mae, epoch)
        writer.add_scalar(f"sub-{subject}/test/r", test_r, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("BT", batch_time, epoch)
        writer.add_scalar("DT", data_time, epoch)
        writer.add_scalar("epoch", epoch, epoch)
        print(f"epoch {epoch}, total time {t2 - start_time:.2f}, epoch time {t2 - t1:.3f} "
              f"loss {loss:.4f} train r {corr:.4f} - test MAE {test_mae:.4f} test r {test_r:.4f}")

        save_model(model, None, None, opts, epoch, save_file)

def main():
    opts = parse_args()
    util.set_seed(opts.trial)

    run_name = (f"predictor_{opts.model}_w{opts.stimulus_window}_{opts.optimizer}_lr{opts.lr}_"
                f"decay{opts.lr_decay}_wd{opts.weight_decay}_bsz{opts.batch_size}_dropout{opts.dropout}_"
                f"epochs{opts.epochs}_s{opts.trial}")

    tb_dir = os.path.join(opts.log_dir, "tensorboard", run_name)
    save_dir = os.path.join(opts.features_dir, "predictors", run_name)
    opts.save_dir = save_dir
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print("Saving weights to", save_dir)
    print("Saving logs to", tb_dir)

    wandb.init(project="algonauts-challenge-2025", name=run_name, config=opts, sync_tensorboard=True)
    writer = torch.utils.tensorboard.SummaryWriter(tb_dir)

    packages = util.get_packages_versions()
    wandb.config.update({"env": packages})
    print("Packages:")
    for k, v in packages.items():
        print(f"{k}=={v}")

    for subject in [1,2,3,5]:
        print(f"Training subject {subject}")
        run_training(opts, subject, writer)


if __name__ == '__main__':
    main()


