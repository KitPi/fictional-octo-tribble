import argparse
import csv
import os
import random

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

LR = 5e-4
EPOCHS = 1000
EPOCHS_PER_UPDATE = 1
RUNNAME = "Sen1Floods11"


# ── Dataset ───────────────────────────────────────────────────────────────────


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)


def processAndAugment(data, norm):
    (x, y) = data
    im, label = x.copy(), y.copy()

    im1 = Image.fromarray(im[0])
    im2 = Image.fromarray(im[1])
    label = Image.fromarray(label.squeeze())

    i, j, h, w = transforms.RandomCrop.get_params(im1, (256, 256))
    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    label = F.crop(label, i, j, h, w)

    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        label = F.vflip(label)

    im = torch.stack(
        [transforms.ToTensor()(im1).squeeze(), transforms.ToTensor()(im2).squeeze()]
    )
    im = norm(im)
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(0.003) * label.lt(0.004)):
        label *= 255
    label = label.round()
    return im, label


def processTestIm(data, norm):
    (x, y) = data
    im, label = x.copy(), y.copy()

    im_c1 = Image.fromarray(im[0]).resize((512, 512))
    im_c2 = Image.fromarray(im[1]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))

    crops = [
        (0, 0, 256, 256),
        (0, 256, 256, 256),
        (256, 0, 256, 256),
        (256, 256, 256, 256),
    ]
    im_c1s = [F.crop(im_c1, *c) for c in crops]
    im_c2s = [F.crop(im_c2, *c) for c in crops]
    labels = [F.crop(label, *c) for c in crops]

    ims = torch.stack(
        [
            torch.stack(
                [
                    transforms.ToTensor()(c1).squeeze(),
                    transforms.ToTensor()(c2).squeeze(),
                ]
            )
            for c1, c2 in zip(im_c1s, im_c2s)
        ]
    )
    ims = norm(ims)
    labels = torch.stack([transforms.ToTensor()(lbl).squeeze() for lbl in labels])
    if torch.sum(labels.gt(0.003) * labels.lt(0.004)):
        labels *= 255
    labels = labels.round()
    return ims, labels


# ── Data loading ──────────────────────────────────────────────────────────────


def getArrFlood(fname):
    return rasterio.open(fname).read()


def download_flood_water_data_from_list(file_list, root=None):
    data = []
    for i, (im_fname, mask_fname) in enumerate(file_list):
        if root:
            im_fname = os.path.join(root, im_fname)
            mask_fname = os.path.join(root, mask_fname)
        if not os.path.exists(im_fname) or not os.path.exists(mask_fname):
            continue
        arr_x = np.nan_to_num(getArrFlood(im_fname))
        arr_y = getArrFlood(mask_fname)
        arr_y[arr_y == -1] = 255
        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51
        if i % 100 == 0:
            print(f"  loaded {i}: {os.path.basename(im_fname)}")
        data.append((arr_x, arr_y))
    return data


def load_data_from_csv(csv_path, input_root, label_root):
    files = []
    with open(csv_path) as f:
        for line in csv.reader(f):
            files.append((line[0], line[1]))
    return download_flood_water_data_from_list(
        files,
        root=os.path.commonpath([input_root, label_root])
        if os.path.commonpath([input_root, label_root])
        else None,
    )


def load_flood_data(split, data_dir):
    csv_path = os.path.join("splits", f"flood_{split}_data.csv")
    input_root = os.path.join(data_dir, "S1")
    label_root = os.path.join(data_dir, "Labels")
    files = []
    with open(csv_path) as f:
        for line in csv.reader(f):
            files.append(
                (os.path.join(input_root, line[0]), os.path.join(label_root, line[1]))
            )
    return download_flood_water_data_from_list(files)


# ── Model ─────────────────────────────────────────────────────────────────────


def convertBNtoGN(module, num_groups=16):
    if isinstance(module, nn.modules.batchnorm.BatchNorm2d):
        mod = nn.GroupNorm(
            num_groups, module.num_features, eps=module.eps, affine=module.affine
        )
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
        return mod
    for name, child in module.named_children():
        module.add_module(name, convertBNtoGN(child, num_groups=num_groups))
    return module


def build_model():
    net = models.segmentation.fcn_resnet50(
        pretrained=False, num_classes=2, pretrained_backbone=False
    )
    net.backbone.conv1 = nn.Conv2d(
        2, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    net = convertBNtoGN(net)
    return net


# ── Metrics ───────────────────────────────────────────────────────────────────


def computeIOU(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    intersection = torch.sum(output * target)
    union = torch.sum(target) + torch.sum(output) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou if iou == iou else torch.tensor(0.0).float().cuda()


def computeAccuracy(output, target):
    output = torch.argmax(output, dim=1).flatten()
    target = target.flatten()
    no_ignore = target.ne(255).cuda()
    output = output.masked_select(no_ignore)
    target = target.masked_select(no_ignore)
    correct = torch.sum(output.eq(target))
    return correct.float() / len(target)


# ── Training / Validation ─────────────────────────────────────────────────────


def train_loop(inputs, labels, net, criterion, optimizer, scheduler):
    optimizer.zero_grad()
    net = net.cuda()
    outputs = net(inputs.cuda())
    loss = criterion(outputs["out"], labels.long().cuda())
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss, outputs["out"]


def validate(valid_loader, net, criterion):
    net = net.eval().cuda()
    count = 0
    iou = 0.0
    loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = net(images.cuda())
            loss += criterion(outputs["out"], labels.long().cuda())
            iou += computeIOU(outputs["out"], labels.cuda())
            accuracy += computeAccuracy(outputs["out"], labels.cuda())
            count += 1
    return loss / count, iou / count, accuracy / count


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args):
    writer = SummaryWriter(log_dir=args.log_dir)
    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

    print("Loading data...")
    train_data = load_flood_data("train", args.data_dir)
    valid_data = load_flood_data("valid", args.data_dir)
    print(f"Train samples: {len(train_data)}, Valid samples: {len(valid_data)}")

    train_dataset = InMemoryDataset(train_data, lambda d: processAndAugment(d, norm))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    valid_dataset = InMemoryDataset(valid_data, lambda d: processTestIm(d, norm))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: (
            torch.cat([a[0] for a in x], 0),
            torch.cat([a[1] for a in x], 0),
        ),
    )

    print("Building model...")
    net = build_model()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1, 8]).float().cuda(), ignore_index=255
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, len(train_loader) * 10, T_mult=2, eta_min=0
    )

    start_epoch = 0
    max_valid_iou = 0.0
    global_step = 0

    # resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        max_valid_iou = ckpt.get("max_valid_iou", 0.0)
        print(f"Resumed at epoch {start_epoch}, best val IoU: {max_valid_iou:.4f}")
        net = net.to("cuda")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        net = net.train()
        train_loss = 0.0
        train_iou = 0.0
        train_acc = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for inputs, labels in pbar:
            loss, outputs = train_loop(
                inputs, labels, net, criterion, optimizer, scheduler
            )
            iou = computeIOU(outputs, labels.cuda())
            acc = computeAccuracy(outputs, labels.cuda())

            train_loss += loss.item()
            train_iou += iou.item()
            train_acc += acc.item()
            train_count += 1
            global_step += 1

            pbar.set_postfix(loss=loss.item(), iou=iou.item(), acc=acc.item())

            # log every N steps
            if global_step % args.log_interval == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/iou", iou.item(), global_step)
                writer.add_scalar("train/accuracy", acc.item(), global_step)

        avg_train_loss = train_loss / train_count
        avg_train_iou = train_iou / train_count
        avg_train_acc = train_acc / train_count

        # validate
        val_loss, val_iou, val_acc = validate(valid_loader, net, criterion)

        # tensorboard
        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch)
        writer.add_scalar("epoch/train_iou", avg_train_iou, epoch)
        writer.add_scalar("epoch/train_accuracy", avg_train_acc, epoch)
        writer.add_scalar("epoch/val_loss", val_loss.item(), epoch)
        writer.add_scalar("epoch/val_iou", val_iou.item(), epoch)
        writer.add_scalar("epoch/val_accuracy", val_acc.item(), epoch)
        writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"\nEpoch {epoch:3d} | "
            f"train loss: {avg_train_loss:.4f} iou: {avg_train_iou:.4f} acc: {avg_train_acc:.4f} | "
            f"val loss: {val_loss.item():.4f} iou: {val_iou.item():.4f} acc: {val_acc.item():.4f}"
        )

        # save checkpoint
        ckpt_dict = {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "train_iou": avg_train_iou,
            "val_loss": val_loss.item(),
            "val_iou": val_iou.item(),
            "max_valid_iou": max_valid_iou,
        }
        torch.save(ckpt_dict, os.path.join(args.checkpoint_dir, f"{RUNNAME}_latest.cp"))

        if val_iou > max_valid_iou:
            max_valid_iou = val_iou
            save_path = os.path.join(
                args.checkpoint_dir, f"{RUNNAME}_{epoch}_{val_iou.item():.4f}.cp"
            )
            torch.save(net.state_dict(), save_path)
            print(f"  ** New best model saved: {save_path}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FloodMask model")
    parser.add_argument("--data-dir", default="files", help="Root data directory")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Where to save model checkpoints",
    )
    parser.add_argument("--log-dir", default="runs", help="TensorBoard log directory")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Steps between TensorBoard logs"
    )
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()
    main(args)
