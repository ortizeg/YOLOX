#!/usr/bin/env python3
"""Opt-in training speed-ups, one flag per change, so every change can be tested against the original code path.

Enable with the exp attribute ``perf`` (comma-separated names, or ``all`` for every tier-A/B flag), e.g.
``python -m yolox.tools.train -f exp.py ... perf all`` or ``perf no_empty_cache,simota_batch``.
With ``perf`` empty (the default) every code path is the original YOLOX one.

Tiers (docs/perf/LEDGER.md): A = bit-identical outputs, B = numerically equivalent (tolerance), C = changes results\n(never in ``all``; enable by name and back it with an accuracy run).
"""

from __future__ import annotations

import os

TIER = {
    # GPU step
    "no_empty_cache": "A",  # G1: no torch.cuda.empty_cache() per image in the loss loop
    "nondeterministic": "B",  # G2: keep cudnn.benchmark, do not force cudnn.deterministic when seeded
    "simota_batch": "B",  # G3+G5+G6 (CPU bit-identical; CUDA topk/argmin tie order can differ): SimOTA for the whole batch at once, no per-image host syncs
    "cls_cost_algebra": "B",  # G4: pairwise class cost without the [gt x anchor x class] tensor
    "ema_foreach": "A",  # G7: EMA update with torch._foreach_* ops
    "fused_optim": "B",  # G8: fused AdamW / SGD kernels
    "channels_last": "B",  # G9: NHWC memory format
    "compile": "B",  # G10: torch.compile on backbone + neck
    "ddp_bucket_view": "A",  # G12: DDP gradient_as_bucket_view
    "async_ckpt": "A",  # checkpoints: copy to CPU, write from a background thread (same weights on disk)
    "eval_persistent": "A",  # val dataloader keeps its workers between evaluations
    "eval_amp": "B",  # val inference under fp16 autocast
    # data
    "uint8_transport": "A",  # D1: workers return uint8, cast on GPU
    "skip_noop_resize": "A",  # D2: no identity resize / unneeded copies
    "hsv_lut": "A",  # D3: HSV jitter with lookup tables
    "int_mixup": "A",  # D4: integer mixup blend
    "anno_cache": "A",  # annotations parsed once into memory-mapped arrays; COCO object loaded only if asked for
    "class_map": "A",  # D10: dict class-id lookup at dataset start-up
    # model
    "sppf": "B",  # M1: SPP 5/9/13 as chained 5x5 pools (forward exact; max-pool backward ties route differently)
    "focus_conv": "B",  # M2: Focus as one 6x6 stride-2 conv with rearranged weights
    # pass 3
    "cv_flip": "A",  # P1: horizontal flips with cv2.flip (contiguous copy) instead of a negative-stride numpy view
    "preproc_nopad": "A",  # P2: preproc skips the 114 canvas when the image already fills the input size
    "hwc_transport": "A",  # P3: workers return HWC uint8 (no CHW transpose); the trainer permutes on the GPU
    "int_indices": "A",  # P4: InfiniteSampler yields Python ints, not 0-d tensors (no shared-memory pickling)
    "fast_anno": "A",  # P5: annotation parsing with Python min/max instead of np.max/np.min on tuples
    "simota_nosync": "B",  # P6: batched SimOTA with 2 host syncs per step instead of ~9 (needs simota_batch)
    "resize_nosync": "A",  # P7: multiscale size without a CUDA tensor/.item() (gloo broadcast, no NCCL barrier)
    # pass 4 (tier C: changes the training data distribution, needs accuracy runs)
    "decode_reuse": "C",  # Q3: mosaic/mixup partners drawn from a per-worker pool of recently decoded images
    # pass 5
    "host_label_counts": "A",  # N5: per-image label counts from the CPU targets, not a GPU .tolist() sync in the loss
    "fused_optim_amp": "B",  # N3+N4: rebuild SGD/Adam/AdamW with fused=True in the constructor (sync-free GradScaler step)
    "head_nosync": "A",  # N2: head grids/stride rows built and cached on the device (per level and size), no H2D copies
    "worker_cv2_threads": "A",  # N1: cv2.setNumThreads(0) + no OpenCL in spawned DDP ranks and every dataloader worker
}

_FLAGS: set[str] = set()


def parse(spec: str | None) -> set[str]:
    if not spec:
        return set()
    names = {s.strip() for s in str(spec).split(",") if s.strip()}
    if "all" in names:  # tier A/B only; tier-C flags must be named explicitly
        names = (names - {"all"}) | {k for k, t in TIER.items() if t != "C"}
    if "tierA" in names:
        names = (names - {"tierA"}) | {k for k, t in TIER.items() if t == "A"}
    unknown = names - set(TIER)
    if unknown:
        raise ValueError(f"unknown perf flags: {sorted(unknown)}; known: {sorted(TIER)}")
    return names


def set_flags(spec) -> set[str]:
    """Set the process-wide flags (inherited by forked dataloader workers). Also exported to the environment so
    spawned workers and subprocesses see the same set."""
    global _FLAGS
    _FLAGS = parse(spec) if not isinstance(spec, (set, frozenset)) else set(spec)
    os.environ["YOLOX_PERF"] = ",".join(sorted(_FLAGS))
    return set(_FLAGS)


def on(name: str) -> bool:
    assert name in TIER, name
    return name in _FLAGS


def flags() -> set[str]:
    return set(_FLAGS)


def label_counts(targets):
    """``host_label_counts``: per-image label counts of a CPU target batch [B, M, 5] by the head's rule (row sum > 0),
    as a list of Python ints. Returns None (the head then counts on the device) when any entry is negative: with
    non-negative entries a row sum is > 0 exactly when some entry is > 0, which the trainer's dtype cast and the
    multiscale rescale (a positive factor on the coordinates) cannot change, so the host counts equal the device
    counts bit for bit. YOLOX targets (class id, clipped cx, cy, w, h; zero padding) are always non-negative."""
    if targets.device.type != "cpu" or targets.dim() != 3 or bool((targets < 0).any()):
        return None
    return (targets.sum(dim=2) > 0).sum(dim=1).tolist()


def set_label_counts(model, counts):
    """``host_label_counts``: hand this step's host counts to the model's head (consumed by its next training
    forward). Unwraps DDP/DataParallel (``.module``) and torch.compile (``._orig_mod``). Each DDP rank calls this with
    its own batch's counts."""
    while not hasattr(model, "head"):
        model = model._orig_mod if hasattr(model, "_orig_mod") else model.module
    model.head.host_label_counts = None if counts is None else list(counts)


# picked up at import in subprocesses / spawned workers
if os.environ.get("YOLOX_PERF"):
    _FLAGS = parse(os.environ["YOLOX_PERF"])
