#!/usr/bin/env python3
"""Compare YOLOX-S vs DINOX-S training for 10 epochs on COCO.

Logs per-epoch loss breakdown and timing for both heads, then prints a
side-by-side comparison table. Writes results to CSV.

Usage:
    python scripts/compare_dinox_training.py \
        --data-dir /path/to/COCO --epochs 10 --batch-size 64 --fp16
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare YOLOX-S vs DINOX-S training convergence."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("YOLOX_DATADIR"),
        help="COCO data directory (default: $YOLOX_DATADIR)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Total batch size (default: 64)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks/dinox_comparison",
        help="Output directory for CSVs (default: benchmarks/dinox_comparison)",
    )
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 AMP")
    parser.add_argument("--bf16", action="store_true", help="Enable BF16 AMP")
    return parser


def train_one_config(
    exp_name: str,
    data_dir: Path,
    epochs: int,
    batch_size: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> list[dict[str, float]]:
    """Train a single experiment config and return per-epoch metrics."""
    from yolox.data import DataPrefetcher
    from yolox.exp import get_exp
    from yolox.utils import ModelEMA

    exp = get_exp(exp_name=exp_name)
    exp.max_epoch = epochs
    exp.data_dir = str(data_dir)
    exp.eval_interval = epochs + 1  # disable eval
    exp.print_interval = 50
    exp.warmup_epochs = min(exp.warmup_epochs, max(1, epochs // 3))
    exp.no_aug_epochs = 0

    model = exp.get_model().cuda()
    optimizer = exp.get_optimizer(batch_size)

    exp.dataset = exp.get_dataset()
    train_loader = exp.get_data_loader(batch_size=batch_size, is_distributed=False)

    base_lr = exp.basic_lr_per_img * batch_size
    iters_per_epoch = len(train_loader)
    lr_scheduler = exp.get_lr_scheduler(base_lr, iters_per_epoch)

    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16 and use_amp))
    ema_model = ModelEMA(model, 0.9998)

    logger.info(
        "[{}] Training {} epochs, {} iters/epoch, batch_size={}",
        exp_name, epochs, iters_per_epoch, batch_size,
    )

    results: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        losses: dict[str, list[float]] = {
            "total": [], "iou": [], "conf": [], "cls": [],
        }
        prefetcher = DataPrefetcher(train_loader)
        epoch_start = time.perf_counter()

        for iter_idx in range(iters_per_epoch):
            inps, targets = prefetcher.next()
            if inps is None:
                prefetcher = DataPrefetcher(train_loader)
                inps, targets = prefetcher.next()
                if inps is None:
                    break

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = model(inps, targets)

            loss = outputs["total_loss"]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_iter = epoch * iters_per_epoch + iter_idx + 1
            lr = lr_scheduler.update_lr(global_iter)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            ema_model.update(model)

            losses["total"].append(loss.item())
            losses["iou"].append(outputs["iou_loss"].item())
            losses["conf"].append(outputs["conf_loss"].item())
            losses["cls"].append(outputs["cls_loss"].item())

            if (iter_idx + 1) % exp.print_interval == 0:
                logger.info(
                    "[{}] E{}/{} I{}/{} - loss:{:.4f} iou:{:.4f} conf:{:.4f} cls:{:.4f}",
                    exp_name, epoch + 1, epochs, iter_idx + 1, iters_per_epoch,
                    loss.item(), outputs["iou_loss"].item(),
                    outputs["conf_loss"].item(), outputs["cls_loss"].item(),
                )

        epoch_time = time.perf_counter() - epoch_start
        if not losses["total"]:
            continue

        row = {
            "epoch": epoch + 1,
            "avg_loss": sum(losses["total"]) / len(losses["total"]),
            "iou_loss": sum(losses["iou"]) / len(losses["iou"]),
            "obj_loss": sum(losses["conf"]) / len(losses["conf"]),
            "cls_loss": sum(losses["cls"]) / len(losses["cls"]),
            "epoch_time_s": epoch_time,
            "lr": lr,
        }
        results.append(row)

        logger.info(
            "[{}] Epoch {}/{} done in {:.1f}s - loss:{:.4f} iou:{:.4f} obj:{:.4f} cls:{:.4f}",
            exp_name, epoch + 1, epochs, epoch_time,
            row["avg_loss"], row["iou_loss"], row["obj_loss"], row["cls_loss"],
        )

    return results


def save_csv(results: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "avg_loss", "iou_loss", "obj_loss", "cls_loss", "epoch_time_s", "lr"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved to {}", path)


def print_comparison(
    yolox_results: list[dict[str, float]],
    dinox_results: list[dict[str, float]],
) -> None:
    sep = "=" * 100
    header = (
        f"{'Epoch':>6} | "
        f"{'YOLOX Loss':>10} {'DINOX Loss':>10} {'Delta':>8} | "
        f"{'YOLOX IoU':>10} {'DINOX IoU':>10} {'Delta':>8} | "
        f"{'YOLOX cls':>10} {'DINOX cls':>10} {'Delta':>8} | "
        f"{'YOLOX t/s':>9} {'DINOX t/s':>9}"
    )

    print(f"\n{sep}")
    print("  YOLOX-S vs DINOX-S Training Comparison")
    print(sep)
    print(header)
    print("-" * len(header))

    for y, d in zip(yolox_results, dinox_results):
        loss_delta = d["avg_loss"] - y["avg_loss"]
        iou_delta = d["iou_loss"] - y["iou_loss"]
        cls_delta = d["cls_loss"] - y["cls_loss"]
        print(
            f"{y['epoch']:>6d} | "
            f"{y['avg_loss']:>10.4f} {d['avg_loss']:>10.4f} {loss_delta:>+8.4f} | "
            f"{y['iou_loss']:>10.4f} {d['iou_loss']:>10.4f} {iou_delta:>+8.4f} | "
            f"{y['cls_loss']:>10.4f} {d['cls_loss']:>10.4f} {cls_delta:>+8.4f} | "
            f"{y['epoch_time_s']:>9.1f} {d['epoch_time_s']:>9.1f}"
        )

    print(sep)

    if len(yolox_results) >= 2 and len(dinox_results) >= 2:
        y_speed = sum(r["epoch_time_s"] for r in yolox_results) / len(yolox_results)
        d_speed = sum(r["epoch_time_s"] for r in dinox_results) / len(dinox_results)
        speed_delta = (d_speed - y_speed) / y_speed * 100
        print(f"\n  Avg epoch time: YOLOX={y_speed:.1f}s  DINOX={d_speed:.1f}s  ({speed_delta:+.1f}%)")

        y_final = yolox_results[-1]["avg_loss"]
        d_final = dinox_results[-1]["avg_loss"]
        print(f"  Final loss:     YOLOX={y_final:.4f}  DINOX={d_final:.4f}  (delta={d_final - y_final:+.4f})")
    print()


def main() -> None:
    args = make_parser().parse_args()

    if args.data_dir is None:
        logger.error("No data directory. Set YOLOX_DATADIR or pass --data-dir.")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("Data directory does not exist: {}", data_dir)
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("CUDA required for training comparison.")
        sys.exit(1)

    use_amp = args.fp16 or args.bf16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    output_dir = Path(args.output_dir)

    # Train YOLOX-S baseline
    logger.info("=" * 60)
    logger.info("  Phase 1: Training YOLOX-S baseline")
    logger.info("=" * 60)
    yolox_results = train_one_config(
        "yolox-s", data_dir, args.epochs, args.batch_size, use_amp, amp_dtype,
    )
    save_csv(yolox_results, output_dir / "yolox_s.csv")

    # Clear GPU memory between runs
    torch.cuda.empty_cache()

    # Train DINOX-S
    logger.info("=" * 60)
    logger.info("  Phase 2: Training DINOX-S (RTMDet improvements)")
    logger.info("=" * 60)
    dinox_results = train_one_config(
        "dinox-s", data_dir, args.epochs, args.batch_size, use_amp, amp_dtype,
    )
    save_csv(dinox_results, output_dir / "dinox_s.csv")

    # Print comparison
    print_comparison(yolox_results, dinox_results)


if __name__ == "__main__":
    main()
