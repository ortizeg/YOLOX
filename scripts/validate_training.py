#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.
"""
Validate that YOLOX training converges correctly by running a short
training run on COCO and logging per-epoch loss to CSV.

Usage:
    python scripts/validate_training.py --data-dir /path/to/COCO --epochs 10
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

from loguru import logger

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short YOLOX training and log per-epoch loss."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("YOLOX_DATADIR"),
        help="COCO data directory (default: $YOLOX_DATADIR env variable)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Total batch size (default: 64)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/training_convergence.csv",
        help="CSV output path (default: benchmarks/training_convergence.csv)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 mixed-precision training",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable BF16 mixed-precision training",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = make_parser().parse_args()

    # ------------------------------------------------------------------
    # Validate data directory
    # ------------------------------------------------------------------
    if args.data_dir is None:
        logger.error(
            "No data directory specified. Set the YOLOX_DATADIR environment "
            "variable or pass --data-dir /path/to/COCO."
        )
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error(
            "Data directory does not exist: {}. Please supply a valid COCO "
            "dataset path via --data-dir or $YOLOX_DATADIR.",
            data_dir,
        )
        sys.exit(1)

    train_images = data_dir / "train2017"
    train_ann = data_dir / "annotations" / "instances_train2017.json"
    if not train_images.is_dir() or not train_ann.is_file():
        logger.error(
            "Expected COCO layout under {}: train2017/ and "
            "annotations/instances_train2017.json not found.",
            data_dir,
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("CUDA is required for training. No CUDA device found.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build experiment
    # ------------------------------------------------------------------
    from yolox.exp import get_exp

    exp = get_exp(exp_name="yolox-s")
    exp.max_epoch = args.epochs
    exp.data_dir = str(data_dir)
    exp.eval_interval = args.epochs + 1  # disable eval during short run
    exp.print_interval = 1
    # Scale warmup to the shorter run
    exp.warmup_epochs = min(exp.warmup_epochs, max(1, args.epochs // 3))
    exp.no_aug_epochs = 0  # keep augmentation on for the full short run

    logger.info("Experiment config:\n{}", exp)

    # ------------------------------------------------------------------
    # Model & optimizer
    # ------------------------------------------------------------------
    model = exp.get_model().cuda()
    optimizer = exp.get_optimizer(args.batch_size)

    # ------------------------------------------------------------------
    # Data loader
    # ------------------------------------------------------------------
    # Use the experiment's own data-loader factory (handles MosaicDetection,
    # InfiniteSampler, YoloBatchSampler, etc.)
    exp.dataset = exp.get_dataset()
    train_loader = exp.get_data_loader(
        batch_size=args.batch_size, is_distributed=False
    )

    # ------------------------------------------------------------------
    # LR scheduler
    # ------------------------------------------------------------------
    base_lr = exp.basic_lr_per_img * args.batch_size
    iters_per_epoch = len(train_loader)
    lr_scheduler = exp.get_lr_scheduler(base_lr, iters_per_epoch)

    logger.info(
        "Training for {} epochs, {} iters/epoch, batch_size={}",
        args.epochs,
        iters_per_epoch,
        args.batch_size,
    )

    # ------------------------------------------------------------------
    # AMP setup
    # ------------------------------------------------------------------
    use_amp = args.fp16 or args.bf16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    from yolox.utils import ModelEMA

    ema_model = ModelEMA(model, 0.9998)

    # ------------------------------------------------------------------
    # Data prefetcher
    # ------------------------------------------------------------------
    from yolox.data import DataPrefetcher

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    results = []

    for epoch in range(args.epochs):
        model.train()
        epoch_total_loss = []
        epoch_iou_loss = []
        epoch_conf_loss = []
        epoch_cls_loss = []
        epoch_lr = []

        prefetcher = DataPrefetcher(train_loader)
        epoch_start = time.perf_counter()

        for iter_idx in range(iters_per_epoch):
            inps, targets = prefetcher.next()
            if inps is None:
                logger.warning(
                    "Prefetcher returned None at epoch {} iter {}; "
                    "reinitialising.",
                    epoch,
                    iter_idx,
                )
                prefetcher = DataPrefetcher(train_loader)
                inps, targets = prefetcher.next()
                if inps is None:
                    break

            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                outputs = model(inps, targets)

            loss = outputs["total_loss"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update LR
            global_iter = epoch * iters_per_epoch + iter_idx + 1
            lr = lr_scheduler.update_lr(global_iter)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Update EMA
            ema_model.update(model)

            # Record
            epoch_total_loss.append(loss.item())
            epoch_iou_loss.append(outputs["iou_loss"].item())
            epoch_conf_loss.append(outputs["conf_loss"].item())
            epoch_cls_loss.append(outputs["cls_loss"].item())
            epoch_lr.append(lr)

            if (iter_idx + 1) % exp.print_interval == 0:
                logger.info(
                    "Epoch [{}/{}] Iter [{}/{}] - "
                    "total_loss: {:.4f}, iou: {:.4f}, conf: {:.4f}, "
                    "cls: {:.4f}, lr: {:.6f}",
                    epoch + 1,
                    args.epochs,
                    iter_idx + 1,
                    iters_per_epoch,
                    loss.item(),
                    outputs["iou_loss"].item(),
                    outputs["conf_loss"].item(),
                    outputs["cls_loss"].item(),
                    lr,
                )

        epoch_time = time.perf_counter() - epoch_start

        if not epoch_total_loss:
            logger.warning("No iterations completed for epoch {}", epoch + 1)
            continue

        avg_total = sum(epoch_total_loss) / len(epoch_total_loss)
        avg_iou = sum(epoch_iou_loss) / len(epoch_iou_loss)
        avg_conf = sum(epoch_conf_loss) / len(epoch_conf_loss)
        avg_cls = sum(epoch_cls_loss) / len(epoch_cls_loss)
        avg_lr = sum(epoch_lr) / len(epoch_lr)

        results.append(
            {
                "epoch": epoch + 1,
                "avg_loss": avg_total,
                "iou_loss": avg_iou,
                "obj_loss": avg_conf,
                "cls_loss": avg_cls,
                "lr": avg_lr,
            }
        )

        logger.info(
            "Epoch {}/{} done in {:.1f}s - avg_loss: {:.4f}, "
            "iou: {:.4f}, obj: {:.4f}, cls: {:.4f}, lr: {:.6f}",
            epoch + 1,
            args.epochs,
            epoch_time,
            avg_total,
            avg_iou,
            avg_conf,
            avg_cls,
            avg_lr,
        )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["epoch", "avg_loss", "iou_loss", "obj_loss", "cls_loss", "lr"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Results saved to {}", output_path)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    header = f"{'Epoch':>6} | {'Avg Loss':>10} | {'IoU Loss':>10} | {'Obj Loss':>10} | {'Cls Loss':>10} | {'LR':>12}"
    separator = "-" * len(header)

    print("\n" + separator)
    print("  Training Convergence Summary")
    print(separator)
    print(header)
    print(separator)
    for row in results:
        print(
            f"{row['epoch']:>6d} | "
            f"{row['avg_loss']:>10.4f} | "
            f"{row['iou_loss']:>10.4f} | "
            f"{row['obj_loss']:>10.4f} | "
            f"{row['cls_loss']:>10.4f} | "
            f"{row['lr']:>12.6f}"
        )
    print(separator)

    if len(results) >= 2:
        first_loss = results[0]["avg_loss"]
        last_loss = results[-1]["avg_loss"]
        delta = last_loss - first_loss
        direction = "decreased" if delta < 0 else "increased"
        print(
            f"\n  Loss {direction} from {first_loss:.4f} to {last_loss:.4f} "
            f"(delta: {delta:+.4f})"
        )
    print()


if __name__ == "__main__":
    main()
