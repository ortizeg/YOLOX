#!/usr/bin/env python3
"""GPU-step benchmark and GPU equivalence check for the perf flags (no data loading in the timed loop).

Captures K real training batches once (original data path), then for each flag set rebuilds the model from the
same initial weights and runs the trainer's step (fp16 autocast, GradScaler, optimizer, EMA, multiscale resize
every 10 iterations with a fixed size sequence) and reports seconds per iteration.

  python tools/perf/bench_step.py -f exps/default/yolox_m_basketball_merged5_adamw.py -c yolox_m.pth \
      --sets "" no_empty_cache simota_batch ... --out bench_step.json  [--check]  [exp opts...]

--check: deterministic algorithms; every set's per-iteration losses and final weights are compared with the
baseline ("" set, run twice to show it reproduces itself). Tier A sets must be bit-identical.
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import ModelEMA, perf


def capture_batches(exp, k, batch):
    perf.set_flags("")
    loader = exp.get_data_loader(batch_size=batch, is_distributed=False, no_aug=False, cache_img=None)
    it = iter(loader)
    out = []
    for _ in range(k):
        inps, targets, _, _ = next(it)
        out.append((inps.float().cuda(), targets.float().cuda(), perf.label_counts(targets)))
    del it, loader
    return out


def run_set(exp, spec, init_sd, batches, iters, warmup, sizes, check):
    perf.set_flags(spec)
    torch.backends.cudnn.deterministic = check or not perf.on("nondeterministic")
    torch.backends.cudnn.benchmark = not check
    exp.__dict__.pop("model", None)
    exp.__dict__.pop("optimizer", None)
    torch.manual_seed(0)
    model = exp.get_model()
    model.load_state_dict(init_sd)
    model.cuda().train()
    cl = perf.on("channels_last")
    if cl:
        model.to(memory_format=torch.channels_last)
    opt = exp.get_optimizer(16)
    if perf.on("fused_optim") and isinstance(opt, (torch.optim.Adam, torch.optim.AdamW)):
        for g in opt.param_groups:
            g["fused"], g["foreach"] = True, False
    for g in opt.param_groups:
        g["lr"] = 1e-4
    ema = ModelEMA(model, 0.9998)
    if perf.on("compile"):
        from torch import _dynamo
        _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 32)
        model.backbone.compile(dynamic=False)
    scaler = torch.amp.GradScaler("cuda")
    losses, times = [], []
    for i in range(warmup + iters):
        inps, targets, counts = batches[i % len(batches)]
        inps, targets = inps.clone().half(), targets.clone().half()
        if perf.on("host_label_counts"):  # the trainer's prefetcher computes these from the CPU batch
            perf.set_label_counts(model, counts)
        tsize = sizes[(i // 10) % len(sizes)]
        inps, targets = exp.preprocess(inps, targets, (tsize, tsize))
        if cl:
            inps = inps.contiguous(memory_format=torch.channels_last)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(inps, targets)
        loss = out["total_loss"]
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
        losses.append(float(loss))
    final = {k: v.detach().float().cpu() for k, v in ema.ema.state_dict().items()}
    return losses, times, final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--exp", required=True)
    p.add_argument("-c", "--ckpt")
    p.add_argument("--sets", nargs="+", required=True, help="flag sets; end the list with -- before exp opts")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--iters", type=int, default=110)
    p.add_argument("--warmup", type=int, default=110)
    p.add_argument("--check", action="store_true")
    p.add_argument("--ref", default="", help="--check reference set (default: baseline); e.g. the validated set, "
                   "to show a new flag is bit-identical on top of it")
    p.add_argument("--out", required=True)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.opts and a.opts[0] == "--":
        a.opts = a.opts[1:]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if a.check:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    exp = get_exp(a.exp, None)
    exp.merge(a.opts)
    exp.data_num_workers = 4
    base = exp.input_size[0] // 32
    r = getattr(exp, "multiscale_range", 5)
    sizes = [32 * s for s in [base + ((j * 7) % (2 * r + 1)) - r for j in range(2 * r + 1)]]
    batches = capture_batches(exp, a.k, a.batch)
    model = exp.get_model()
    if a.ckpt:
        from yolox.utils import load_ckpt
        load_ckpt(model, torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"])
    init_sd = copy.deepcopy(model.state_dict())
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch": a.batch,
           "sizes": sizes, "iters": a.iters, "warmup": a.warmup, "check": a.check, "sets": {}}
    sets = list(a.sets)
    sets = [""] + [x for x in sets if x != ""]
    if a.check:
        sets = [a.ref] + sets  # reference first (twice when it is the baseline): shows the check is meaningful
    ref = None
    for spec in sets:
        try:
            losses, times, final = run_set(exp, spec, init_sd, batches, a.iters, a.warmup, sizes, a.check)
        except Exception as e:  # one failing flag set must not lose the others
            logger.exception(f"set {spec!r} failed")
            res["sets"][spec or "baseline"] = {"flags": sorted(perf.parse(spec)), "error": repr(e)[:300]}
            torch.cuda.empty_cache()
            continue
        row = {"flags": sorted(perf.parse(spec)), "sec_per_iter_median": float(np.median(times)),
               "sec_per_iter_mean": float(np.mean(times)), "img_per_s": a.batch / float(np.median(times))}
        if a.check:
            if ref is None:
                ref = (losses, final)
            else:
                rl, rf = ref
                row["losses_identical"] = losses == rl
                row["loss_max_rel_diff"] = max(abs(x - y) / max(abs(y), 1e-12) for x, y in zip(losses, rl))
                row["weights_identical"] = all(torch.equal(final[k], rf[k]) for k in rf)
                row["weight_max_abs_diff"] = max(float((final[k] - rf[k]).abs().max()) for k in rf)
        key = spec or "baseline"
        if a.check and a.ref and ref is not None and losses is ref[0]:
            key = "ref:" + key  # the reference run; the default (baseline) keeps the old "baseline" / "baseline'" keys
        while key in res["sets"]:
            key += "'"
        res["sets"][key] = row
        logger.info(f"{key:<40} {row['sec_per_iter_median'] * 1000:8.1f} ms/iter  "
                    + " ".join(f"{k}={v}" for k, v in row.items() if k.startswith(("losses", "weights", "loss_max"))))
    b = res["sets"]["baseline"]["sec_per_iter_median"]
    for k, row in res["sets"].items():
        if "sec_per_iter_median" in row:
            row["speedup_vs_baseline"] = b / row["sec_per_iter_median"]
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
