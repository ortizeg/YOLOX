#!/usr/bin/env python3
"""Validate YOLOX inference accuracy on COCO val2017.

Runs a pretrained YOLOX-S model through the standard COCO evaluator and
reports mAP metrics.  Results are persisted to a JSON file so downstream
CI or benchmarking pipelines can consume them.

Usage:
    python scripts/validate_inference.py \
        --ckpt weights/yolox_s.pth \
        --data-dir datasets/COCO
"""

import argparse
import json
import os
import sys
import time

import torch
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import fuse_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate YOLOX inference accuracy on COCO val2017",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root directory of the COCO dataset (contains annotations/, val2017/, etc.)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a YOLOX-S checkpoint file (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/inference_validation.json",
        help="Path where the JSON results will be written (default: benchmarks/inference_validation.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used during evaluation (default: 64)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run inference in FP16 (half-precision) mode",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Fuse Conv + BN layers before evaluation for faster inference",
    )
    parser.add_argument(
        "--conf-thre",
        type=float,
        default=0.01,
        help="Confidence score threshold (default: 0.01)",
    )
    parser.add_argument(
        "--nms-thre",
        type=float,
        default=0.65,
        help="NMS IoU threshold (default: 0.65)",
    )
    return parser.parse_args()


def _validate_paths(args):
    """Fail fast when required files or directories are missing."""
    if not os.path.isfile(args.ckpt):
        logger.error(f"Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    if args.data_dir is not None and not os.path.isdir(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)


def main():
    args = parse_args()
    _validate_paths(args)

    if not torch.cuda.is_available():
        logger.error("CUDA is required for evaluation but no GPU was detected.")
        sys.exit(1)

    # ---- experiment config ---------------------------------------------------
    exp = get_exp(exp_name="yolox-s")
    if args.data_dir is not None:
        exp.data_dir = args.data_dir
    exp.test_conf = args.conf_thre
    exp.nmsthre = args.nms_thre

    # ---- model ---------------------------------------------------------------
    model = exp.get_model()

    logger.info(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    logger.info("Checkpoint loaded successfully.")

    model.cuda()
    model.eval()

    if args.fuse:
        logger.info("Fusing Conv + BN layers ...")
        model = fuse_model(model)

    # ---- evaluation ----------------------------------------------------------
    evaluator = exp.get_evaluator(args.batch_size, is_distributed=False)

    logger.info("Starting COCO evaluation ...")
    start_time = time.time()

    ap50_95, ap50, summary = evaluator.evaluate(
        model, distributed=False, half=args.fp16
    )

    elapsed = time.time() - start_time
    logger.info(f"\n{summary}")
    logger.info(f"Evaluation completed in {elapsed:.1f}s")

    # ---- persist results -----------------------------------------------------
    results = {
        "model": "yolox-s",
        "checkpoint": os.path.abspath(args.ckpt),
        "mAP_50_95": float(ap50_95),
        "mAP_50": float(ap50),
        "conf_thre": args.conf_thre,
        "nms_thre": args.nms_thre,
        "fp16": args.fp16,
        "fuse": args.fuse,
        "batch_size": args.batch_size,
        "elapsed_seconds": round(elapsed, 2),
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
