#!/usr/bin/env python3
"""Where does one training step spend its time? Stage breakdown (CUDA events) for the original code and a flag
set, plus torch.profiler's top CUDA ops for the flag set.
  python tools/perf/profile_step.py -f <exp> -c yolox_m.pth --flags <set> --out profile.json -- [exp opts]
Stages: backbone+neck forward, head forward + SimOTA + loss, backward, optimizer (scaler step/update), EMA.
"""

import argparse
import copy
import json

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from yolox.exp import get_exp
from yolox.utils import ModelEMA, load_ckpt, perf

import sys, os  # noqa: E401
sys.path.insert(0, os.path.dirname(__file__))
from bench_step import capture_batches  # noqa: E402


def build(exp, spec, init_sd):
    perf.set_flags(spec)
    torch.backends.cudnn.deterministic = not perf.on("nondeterministic")
    torch.backends.cudnn.benchmark = True
    exp.__dict__.pop("model", None)
    exp.__dict__.pop("optimizer", None)
    model = exp.get_model()
    model.load_state_dict(init_sd)
    model.cuda().train()
    if perf.on("channels_last"):
        model.to(memory_format=torch.channels_last)
    opt = exp.get_optimizer(16)
    if perf.on("fused_optim"):
        for g in opt.param_groups:
            g["fused"], g["foreach"] = True, False
    for g in opt.param_groups:
        g["lr"] = 1e-4
    return model, opt, ModelEMA(model, 0.9998), torch.amp.GradScaler("cuda")


def step(exp, model, opt, ema, scaler, inps, targets, tsize, ev=None):
    inps, targets = exp.preprocess(inps.clone().half(), targets.clone().half(), (tsize, tsize))
    if perf.on("channels_last"):
        inps = inps.contiguous(memory_format=torch.channels_last)
    mark = (lambda i: ev[i].record()) if ev else (lambda i: None)
    mark(0)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        fpn = model.backbone(inps)
        mark(1)
        loss = model.head(fpn, targets, inps)[0]
    mark(2)
    opt.zero_grad()
    scaler.scale(loss).backward()
    mark(3)
    scaler.step(opt)
    scaler.update()
    mark(4)
    ema.update(model)
    mark(5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--exp", required=True)
    p.add_argument("-c", "--ckpt", required=True)
    p.add_argument("--flags", required=True)
    p.add_argument("--iters", type=int, default=110)
    p.add_argument("--out", required=True)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    opts = a.opts[1:] if a.opts[:1] == ["--"] else a.opts
    exp = get_exp(a.exp, None)
    exp.merge(opts)
    base = exp.input_size[0] // 32
    r = exp.multiscale_range
    sizes = [32 * s for s in [base + ((j * 7) % (2 * r + 1)) - r for j in range(2 * r + 1)]]
    batches = capture_batches(exp, 8, 16)
    m0 = exp.get_model()
    load_ckpt(m0, torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"])
    init_sd = copy.deepcopy(m0.state_dict())
    names = ["backbone+neck fwd", "head fwd + SimOTA + loss", "backward", "optimizer", "EMA"]
    res = {"gpu": torch.cuda.get_device_name(0), "sizes": sizes, "stages": {}}
    for label, spec in [("original", ""), ("speed-ups", a.flags)]:
        model, opt, ema, scaler = build(exp, spec, init_sd)
        for i in range(110):  # warm up every size (cudnn.benchmark autotune)
            step(exp, model, opt, ema, scaler, *batches[i % 8], sizes[(i // 10) % len(sizes)])
        acc = np.zeros(5)
        for i in range(a.iters):
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            step(exp, model, opt, ema, scaler, *batches[i % 8], sizes[(i // 10) % len(sizes)], ev)
            torch.cuda.synchronize()
            acc += [ev[k].elapsed_time(ev[k + 1]) for k in range(5)]
        ms = acc / a.iters
        res["stages"][label] = {n: round(float(v), 2) for n, v in zip(names, ms)} | {"total": round(float(ms.sum()), 2)}
        print(label, res["stages"][label])  # noqa: T201
        if label == "speed-ups":
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                for i in range(20):
                    step(exp, model, opt, ema, scaler, *batches[i % 8], sizes[(i // 10) % len(sizes)])
                torch.cuda.synchronize()
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
            res["top_ops"] = table
            print(table)  # noqa: T201
        del model, opt, ema
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
