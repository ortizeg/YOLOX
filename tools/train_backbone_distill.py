#!/usr/bin/env python3
"""Train CSPDarknet backbone with DINOv2 distillation on ImageNet.

Usage:
    python tools/train_backbone_distill.py \
        --data-dir /path/to/imagenet \
        --epochs 100 \
        --batch-size 256 \
        --lr 0.001 \
        --output-dir backbone_distill_output

The script produces a backbone checkpoint that can be loaded into YOLOX
for detection fine-tuning:
    python tools/train.py -f exp.py -c backbone_distill_output/best_backbone.pth
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from loguru import logger

from yolox.models.darknet import CSPDarknet
from yolox.models.backbone_distill import BackboneDistillModel


def make_parser():
    parser = argparse.ArgumentParser("Backbone Distillation Training")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="ImageNet directory (with train/ and val/ subdirs)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lambda-feat", type=float, default=2.0,
                        help="Feature alignment loss weight")
    parser.add_argument("--lambda-cls-token", type=float, default=1.0,
                        help="CLS token loss weight")
    parser.add_argument("--teacher-model", type=str, default="dinov2_vitb14")
    parser.add_argument("--teacher-layer", type=int, default=12)
    parser.add_argument("--depth", type=float, default=0.33,
                        help="CSPDarknet depth multiplier")
    parser.add_argument("--width", type=float, default=0.50,
                        help="CSPDarknet width multiplier")
    parser.add_argument("--input-size", type=int, default=224,
                        help="ImageNet input size")
    parser.add_argument("--output-dir", type=str, default="backbone_distill_output")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    return parser


def build_dataloaders(args):
    """Build ImageNet train and val dataloaders."""
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images gracefully

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(args.input_size + 32),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"),
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info("Train: {} images, Val: {} images",
                len(train_dataset), len(val_dataset))
    return train_loader, val_loader


def build_model(args):
    """Build backbone distillation model."""
    backbone = CSPDarknet(
        dep_mul=args.depth,
        wid_mul=args.width,
        out_features=("dark3", "dark4", "dark5"),
    )

    student_dim = int(1024 * args.width)  # dark5 channels

    model = BackboneDistillModel(
        backbone=backbone,
        num_classes=1000,
        teacher_model=args.teacher_model,
        teacher_layer=args.teacher_layer,
        student_dim=student_dim,
        lambda_feat=args.lambda_feat,
        lambda_cls_token=args.lambda_cls_token,
    )
    return model


def build_optimizer(model, args):
    """Build AdamW optimizer with separate param groups."""
    # No weight decay for BN and bias
    decay_params = []
    no_decay_params = []
    teacher_excluded = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            teacher_excluded.append(name)
            continue
        if "bn" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    logger.info("Optimizer: {} decay params, {} no-decay params, {} frozen teacher params",
                len(decay_params), len(no_decay_params), len(teacher_excluded))
    return optimizer


def cosine_lr_schedule(optimizer, epoch, max_epoch, warmup_epochs, base_lr, min_lr=1e-6):
    """Cosine LR with linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (max_epoch - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + __import__("math").cos(progress * 3.14159))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_feat_loss = 0
    total_acc = 0
    num_batches = 0

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.fp16):
            outputs = model(images, targets)
            loss = outputs["total_loss"]

        optimizer.zero_grad()
        if args.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += outputs["total_loss"].item()
        total_cls_loss += outputs["cls_loss"].item()
        total_feat_loss += outputs["feat_loss"].item()
        total_acc += outputs["acc1"].item()
        num_batches += 1

        if (i + 1) % 100 == 0:
            logger.info(
                "epoch {}/{}, iter {}/{}: loss={:.3f}, cls={:.3f}, feat={:.3f}, "
                "cls_tok={:.3f}, acc1={:.1%}, lr={:.6f}",
                epoch + 1, args.epochs, i + 1, len(loader),
                outputs["total_loss"].item(),
                outputs["cls_loss"].item(),
                outputs["feat_loss"].item(),
                outputs["cls_token_loss"].item(),
                outputs["acc1"].item(),
                optimizer.param_groups[0]["lr"],
            )

    return {
        "loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "feat_loss": total_feat_loss / num_batches,
        "acc1": total_acc / num_batches,
    }


@torch.no_grad()
def validate(model, loader, device, args):
    """Validate on ImageNet val set."""
    model.eval()
    correct = 0
    correct5 = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.fp16):
            logits = model(images)

        _, pred = logits.topk(5, dim=1)
        correct += (pred[:, 0] == targets).sum().item()
        correct5 += (pred == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)

    acc1 = correct / total
    acc5 = correct5 / total
    return acc1, acc5


def main():
    args = make_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", device)

    # Build model
    model = build_model(args)
    model.to(device)
    logger.info("Model built: backbone params={:,}",
                sum(p.numel() for p in model.backbone.parameters()))

    # Build data
    train_loader, val_loader = build_dataloaders(args)

    # Build optimizer
    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    # Resume
    start_epoch = 0
    best_acc1 = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_acc1 = ckpt.get("best_acc1", 0)
        logger.info("Resumed from epoch {}, best_acc1={:.1%}", start_epoch, best_acc1)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr_schedule(
            optimizer, epoch, args.epochs, args.warmup_epochs, args.lr,
        )

        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        train_time = time.time() - t0

        logger.info(
            "Epoch {}/{}: loss={:.3f}, cls={:.3f}, feat={:.3f}, acc1={:.1%}, "
            "lr={:.6f}, time={:.0f}s",
            epoch + 1, args.epochs,
            train_metrics["loss"], train_metrics["cls_loss"],
            train_metrics["feat_loss"], train_metrics["acc1"],
            lr, train_time,
        )

        # Validate
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            acc1, acc5 = validate(model, val_loader, device, args)
            logger.info("Val: acc1={:.1%}, acc5={:.1%}", acc1, acc5)

            # Save best
            if acc1 > best_acc1:
                best_acc1 = acc1
                # Save full checkpoint
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc1": best_acc1,
                    "args": vars(args),
                }, os.path.join(args.output_dir, "best_ckpt.pth"))

                # Save backbone-only weights (for YOLOX fine-tuning)
                torch.save(
                    {"model": model.get_backbone_state_dict()},
                    os.path.join(args.output_dir, "best_backbone.pth"),
                )
                logger.info("New best! acc1={:.1%}, saved backbone to {}",
                           acc1, args.output_dir)

        # Save latest
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc1": best_acc1,
            "args": vars(args),
        }, os.path.join(args.output_dir, "latest_ckpt.pth"))

    logger.info("Training complete. Best val acc1={:.1%}", best_acc1)
    logger.info("Backbone weights saved to: {}/best_backbone.pth", args.output_dir)


if __name__ == "__main__":
    main()
