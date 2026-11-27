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
    "lean_ckpt": "A",  # N9: `latest` every exp.save_interval epochs, last_epoch hardlinked, best/epoch_N weights-only
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


# picked up at import in subprocesses / spawned workers
if os.environ.get("YOLOX_PERF"):
    _FLAGS = parse(os.environ["YOLOX_PERF"])
