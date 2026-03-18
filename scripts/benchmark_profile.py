#!/usr/bin/env python3
"""
Comprehensive profiling script for YOLOX.

Measures every component of training and inference using CUDA events
for accurate GPU timing. Outputs results as a formatted table and JSON.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from tabulate import tabulate

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.utils import postprocess


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOX benchmark profiling script")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/profile.json",
        help="Output JSON path (default: benchmarks/profile.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--infer-images",
        type=int,
        default=1000,
        help="Number of inference images (default: 1000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    return parser.parse_args()


def create_model():
    """Create YOLOX-s model with proper initialization."""
    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(0.33, 0.50, in_channels=in_channels)
    head = YOLOXHead(80, 0.50, in_channels=in_channels)
    model = YOLOX(backbone, head)

    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    return model


def create_optimizer(model):
    """Create SGD optimizer with 3 param groups (matching yolox_base.py)."""
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optimizer = torch.optim.SGD(pg0, lr=0.01, momentum=0.9, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": pg2})
    return optimizer


def create_targets(batch_size):
    """Create random targets tensor on CUDA."""
    targets = torch.zeros(batch_size, 50, 5, device="cuda")
    for b in range(batch_size):
        for i in range(5):
            targets[b, i] = torch.tensor([float(i % 80), 320, 320, 100, 100])
    return targets


def cuda_timer():
    """Return a pair of CUDA events for timing."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def profile_training(model, optimizer, batch_size, train_iters, warmup):
    """Profile all training components."""
    print("\n--- Training Profiling ---")
    model.train()
    targets = create_targets(batch_size)

    # ---- Warmup ----
    print(f"  Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        imgs_cpu = torch.randn(batch_size, 3, 640, 640)
        imgs = imgs_cpu.cuda()
        outputs = model(imgs, targets)
        loss = outputs
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # ---- 1. Data loader time (CPU random batch creation) ----
    print(f"  Profiling data loader ({train_iters} iters)...")
    start_evt, end_evt = cuda_timer()
    torch.cuda.synchronize()
    start_evt.record()
    for _ in range(train_iters):
        imgs_cpu = torch.randn(batch_size, 3, 640, 640)
    end_evt.record()
    torch.cuda.synchronize()
    data_loader_ms = start_evt.elapsed_time(end_evt) / train_iters

    # ---- 2. Host-to-GPU transfer ----
    print(f"  Profiling host->GPU transfer ({train_iters} iters)...")
    batches_cpu = [torch.randn(batch_size, 3, 640, 640) for _ in range(train_iters)]
    torch.cuda.synchronize()
    start_evt, end_evt = cuda_timer()
    start_evt.record()
    for i in range(train_iters):
        _ = batches_cpu[i].cuda()
    end_evt.record()
    torch.cuda.synchronize()
    host_to_gpu_ms = start_evt.elapsed_time(end_evt) / train_iters
    del batches_cpu

    # ---- 3. Forward pass ----
    print(f"  Profiling forward pass ({train_iters} iters)...")
    imgs = torch.randn(batch_size, 3, 640, 640, device="cuda")
    torch.cuda.synchronize()
    start_evt, end_evt = cuda_timer()
    start_evt.record()
    for _ in range(train_iters):
        outputs = model(imgs, targets)
    end_evt.record()
    torch.cuda.synchronize()
    forward_pass_ms = start_evt.elapsed_time(end_evt) / train_iters

    # ---- 4. Backward pass ----
    print(f"  Profiling backward pass ({train_iters} iters)...")
    total_backward_ms = 0.0
    for _ in range(train_iters):
        optimizer.zero_grad()
        outputs = model(imgs, targets)
        loss = outputs
        torch.cuda.synchronize()
        s, e = cuda_timer()
        s.record()
        loss.backward()
        e.record()
        torch.cuda.synchronize()
        total_backward_ms += s.elapsed_time(e)
    backward_pass_ms = total_backward_ms / train_iters

    # ---- 5. Optimizer step ----
    print(f"  Profiling optimizer step ({train_iters} iters)...")
    total_optim_ms = 0.0
    for _ in range(train_iters):
        optimizer.zero_grad()
        outputs = model(imgs, targets)
        loss = outputs
        loss.backward()
        torch.cuda.synchronize()
        s, e = cuda_timer()
        s.record()
        optimizer.step()
        e.record()
        torch.cuda.synchronize()
        total_optim_ms += s.elapsed_time(e)
    optimizer_step_ms = total_optim_ms / train_iters

    # ---- 6. Total training throughput ----
    print(f"  Profiling total training throughput ({train_iters} iters)...")
    torch.cuda.synchronize()
    start_evt, end_evt = cuda_timer()
    start_evt.record()
    for _ in range(train_iters):
        imgs_cpu = torch.randn(batch_size, 3, 640, 640)
        imgs_gpu = imgs_cpu.cuda()
        optimizer.zero_grad()
        outputs = model(imgs_gpu, targets)
        loss = outputs
        loss.backward()
        optimizer.step()
    end_evt.record()
    torch.cuda.synchronize()
    total_time_sec = start_evt.elapsed_time(end_evt) / 1000.0
    total_throughput = (train_iters * batch_size) / total_time_sec

    # ---- 7. Peak GPU memory ----
    peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    results = {
        "data_loader_ms": round(data_loader_ms, 3),
        "host_to_gpu_ms": round(host_to_gpu_ms, 3),
        "forward_pass_ms": round(forward_pass_ms, 3),
        "backward_pass_ms": round(backward_pass_ms, 3),
        "optimizer_step_ms": round(optimizer_step_ms, 3),
        "total_throughput_imgs_per_sec": round(total_throughput, 2),
        "peak_gpu_memory_mb": round(peak_gpu_memory_mb, 2),
    }
    return results


def profile_inference(model, infer_images, warmup):
    """Profile inference forward pass and postprocessing."""
    print("\n--- Inference Profiling ---")
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    # ---- Warmup ----
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            img = torch.randn(1, 3, 640, 640, device="cuda")
            _ = model(img)
    torch.cuda.synchronize()

    # ---- Forward pass ----
    print(f"  Profiling inference forward pass ({infer_images} images)...")
    torch.cuda.synchronize()
    start_evt, end_evt = cuda_timer()
    start_evt.record()
    with torch.no_grad():
        for _ in range(infer_images):
            img = torch.randn(1, 3, 640, 640, device="cuda")
            outputs = model(img)
    end_evt.record()
    torch.cuda.synchronize()
    forward_ms_per_img = start_evt.elapsed_time(end_evt) / infer_images

    # ---- Postprocess / NMS ----
    print(f"  Profiling postprocess/NMS ({infer_images} images)...")
    # Generate a representative set of outputs for postprocessing
    with torch.no_grad():
        sample_output = model(torch.randn(1, 3, 640, 640, device="cuda"))

    torch.cuda.synchronize()
    start_evt, end_evt = cuda_timer()
    start_evt.record()
    for _ in range(infer_images):
        _ = postprocess(sample_output, num_classes=80, conf_thre=0.7, nms_thre=0.45)
    end_evt.record()
    torch.cuda.synchronize()
    postprocess_ms_per_img = start_evt.elapsed_time(end_evt) / infer_images

    total_ms_per_img = forward_ms_per_img + postprocess_ms_per_img
    peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    results = {
        "forward_pass_ms_per_img": round(forward_ms_per_img, 3),
        "postprocess_ms_per_img": round(postprocess_ms_per_img, 3),
        "total_ms_per_img": round(total_ms_per_img, 3),
        "peak_gpu_memory_mb": round(peak_gpu_memory_mb, 2),
    }
    return results


def print_results(config, training_results, inference_results):
    """Print results as formatted tables."""
    print("\n" + "=" * 60)
    print("YOLOX Benchmark Profiling Results")
    print("=" * 60)

    # Config table
    config_table = [
        ["Model", config["model"]],
        ["Batch Size", config["batch_size"]],
        ["Input Size", f"{config['input_size'][0]}x{config['input_size'][1]}"],
    ]
    print("\nConfiguration:")
    print(tabulate(config_table, headers=["Parameter", "Value"], tablefmt="grid"))

    # Training table
    train_table = [
        ["Data Loader", f"{training_results['data_loader_ms']:.3f} ms"],
        ["Host -> GPU Transfer", f"{training_results['host_to_gpu_ms']:.3f} ms"],
        ["Forward Pass", f"{training_results['forward_pass_ms']:.3f} ms"],
        ["Backward Pass", f"{training_results['backward_pass_ms']:.3f} ms"],
        ["Optimizer Step", f"{training_results['optimizer_step_ms']:.3f} ms"],
        [
            "Total Throughput",
            f"{training_results['total_throughput_imgs_per_sec']:.2f} imgs/sec",
        ],
        [
            "Peak GPU Memory",
            f"{training_results['peak_gpu_memory_mb']:.2f} MB",
        ],
    ]
    print("\nTraining Profiling:")
    print(tabulate(train_table, headers=["Metric", "Value"], tablefmt="grid"))

    # Inference table
    infer_table = [
        [
            "Forward Pass",
            f"{inference_results['forward_pass_ms_per_img']:.3f} ms/img",
        ],
        [
            "Postprocess/NMS",
            f"{inference_results['postprocess_ms_per_img']:.3f} ms/img",
        ],
        ["Total", f"{inference_results['total_ms_per_img']:.3f} ms/img"],
        [
            "Peak GPU Memory",
            f"{inference_results['peak_gpu_memory_mb']:.2f} MB",
        ],
    ]
    print("\nInference Profiling:")
    print(tabulate(infer_table, headers=["Metric", "Value"], tablefmt="grid"))
    print()


def main():
    args = parse_args()

    assert torch.cuda.is_available(), "CUDA is required for this benchmark script"

    print("YOLOX Benchmark Profiler")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Training iterations: {args.train_iters}")
    print(f"  Inference images: {args.infer_images}")
    print(f"  Warmup iterations: {args.warmup}")

    # Create model
    print("\nCreating YOLOX-s model...")
    model = create_model()
    model.cuda()

    # Create optimizer
    optimizer = create_optimizer(model)

    # Reset memory stats before profiling
    torch.cuda.reset_peak_memory_stats()

    # Profile training
    training_results = profile_training(
        model, optimizer, args.batch_size, args.train_iters, args.warmup
    )

    # Profile inference
    inference_results = profile_inference(model, args.infer_images, args.warmup)

    # Assemble results
    config = {
        "model": "yolox-s",
        "batch_size": args.batch_size,
        "input_size": [640, 640],
    }

    results = {
        "config": config,
        "training": training_results,
        "inference": inference_results,
    }

    # Print results
    print_results(config, training_results, inference_results)

    # Save JSON
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
