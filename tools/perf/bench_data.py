#!/usr/bin/env python3
"""Data-path benchmark: training dataloader throughput (images/s) with the original and the perf data flags.

  python tools/perf/bench_data.py -f <exp> --workers 4 8 --batches 60 --out bench_data.json [exp opts...]
  [--sets "" data "data,cv_flip,preproc_nopad,hwc_transport" -- exp opts...]   ("data" = the validated data flags)
Also times the per-sample pipeline in-process (num_workers 0), which is what one worker core can do.
"""

import argparse
import json
import time

import numpy as np
import torch
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import perf

DATA = "uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map"


def loader_rate(exp, spec, workers, batch, n, warm=10):
    perf.set_flags(spec)
    exp.dataset = None
    exp.data_num_workers = workers
    loader = exp.get_data_loader(batch_size=batch, is_distributed=False, no_aug=False, cache_img=None)
    it = iter(loader)
    for _ in range(warm):
        x, *_ = next(it)
        if workers:
            x.pin_memory()
    t0 = time.perf_counter()
    nbytes = 0
    for _ in range(n):
        x, *_ = next(it)
        nbytes += x.numel() * x.element_size()
    dt = time.perf_counter() - t0
    del it, loader
    return batch * n / dt, nbytes / n


def sample_rate(exp, spec, n=200):
    perf.set_flags(spec)
    exp.dataset = None
    loader = exp.get_data_loader(batch_size=1, is_distributed=False, no_aug=False, cache_img=None)
    ds = loader.dataset
    for i in range(10):
        ds[i]
    t0 = time.perf_counter()
    for i in range(n):
        ds[i % len(ds)]
    return n / (time.perf_counter() - t0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--exp", required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[4, 8])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--batches", type=int, default=60)
    p.add_argument("--out", required=True)
    p.add_argument("--sets", nargs="+", default=["", "data"], help="flag sets; end the list with -- before exp opts")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.opts and a.opts[0] == "--":
        a.opts = a.opts[1:]
    exp = get_exp(a.exp, None)
    exp.merge(a.opts)
    res = {"cpu_count": __import__("os").cpu_count(), "input_size": exp.input_size, "rows": []}
    for spec in [",".join([DATA] + x.split(",")[1:]) if x.split(",")[0] == "data" else x for x in a.sets]:
        sr = sample_rate(exp, spec)
        res["rows"].append({"flags": spec or "baseline", "workers": 0, "img_per_s": sr})
        logger.info(f"{spec or 'baseline':<60} in-process {sr:7.1f} samples/s")
        for w in a.workers:
            r, b = loader_rate(exp, spec, w, a.batch, a.batches)
            res["rows"].append({"flags": spec or "baseline", "workers": w, "img_per_s": r, "bytes_per_batch": b})
            logger.info(f"{spec or 'baseline':<60} workers {w:2d} {r:7.1f} img/s  {b / 2**20:6.1f} MiB/batch")
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
