#!/usr/bin/env python3
"""torch.compile pros and cons for YOLOX training, measured: compile seconds per input size and steady-state ms/iter.

Synthetic batches (no dataset): random images and ~20 random boxes per image, YOLOX-M, batch 16, the full training
step (forward, SimOTA + loss, backward, fused AdamW, foreach EMA) with the validated perf flags. For each variant,
iterates over the multiscale sizes (640-960 = 11 sizes, as YOLOX's multiscale_range 5 around 800), records the
first-iteration time at each size (compile + autotune), then the median of steady iterations per size.
  python tools/perf/compile_bench.py --variants eager bb bb_head bb_maxat dyn --out compile.json
Variants: eager | bb (backbone+neck, dynamic=False) | bb_head (+ head conv towers) |
          bb_maxat (mode="max-autotune-no-cudagraphs") | dyn (dynamic=True, one graph for all sizes)
"""

import argparse
import json
import time

import numpy as np
import torch

from yolox.exp import get_exp
from yolox.utils import ModelEMA, perf

V = ("nondeterministic,no_empty_cache,simota_batch,cls_cost_algebra,ema_foreach,fused_optim,sppf,focus_conv,"
     "channels_last,simota_nosync")


def batch(size, n=16, boxes=20, g=None):
    x = torch.rand(n, 3, size, size, generator=g) * 255
    lab = torch.zeros(n, 120, 5)
    for b in range(n):
        wh = torch.rand(boxes, 2, generator=g) * size * 0.3 + 8
        c = torch.rand(boxes, 2, generator=g) * (size - wh) + wh / 2
        lab[b, :boxes, 0] = torch.randint(0, 5, (boxes,), generator=g).float()
        lab[b, :boxes, 1:3] = c
        lab[b, :boxes, 3:5] = wh
    return x, lab


def run(variant, sizes, iters_per_size):
    perf.set_flags(V)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    from torch import _dynamo
    _dynamo.reset()
    _dynamo.config.cache_size_limit = 64
    exp = get_exp("exps/default/yolox_m_basketball_merged5_adamw.py", None)
    exp.num_classes = 5
    torch.manual_seed(0)
    model = exp.get_model().cuda().train().to(memory_format=torch.channels_last)
    opt = exp.get_optimizer(16)
    for gr in opt.param_groups:
        gr["fused"], gr["foreach"], gr["lr"] = True, False, 1e-4
    ema = ModelEMA(model, 0.9998)
    if variant in ("bb", "bb_head", "bb_maxat", "dyn"):
        kw = {"dynamic": variant == "dyn"}
        if variant == "bb_maxat":
            kw["mode"] = "max-autotune-no-cudagraphs"
        model.backbone.compile(**kw)
        if variant == "bb_head":
            for m in list(model.head.stems) + list(model.head.cls_convs) + list(model.head.reg_convs):
                m.compile(dynamic=False)
    scaler = torch.amp.GradScaler("cuda")
    g = torch.Generator().manual_seed(0)
    data = {s: [tuple(t.cuda() for t in batch(s, g=g)) for _ in range(2)] for s in sizes}
    first, steady = {}, {}
    for s in sizes:
        ts = []
        for i in range(iters_per_size):
            x, lab = data[s][i % 2]
            x = x.half().contiguous(memory_format=torch.channels_last)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(x, lab.half())["total_loss"]
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        first[s] = ts[0]
        steady[s] = float(np.median(ts[5:]))
        print(f"{variant:9} size {s}: first {ts[0]:7.1f} s  steady {steady[s] * 1000:6.1f} ms", flush=True)  # noqa: T201
    return {"first_s": first, "steady_ms": {k: v * 1000 for k, v in steady.items()},
            "compile_total_s": sum(first.values()) - sum(steady.values()) * 1,
            "steady_mean_ms": float(np.mean(list(steady.values()))) * 1000}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["eager", "bb", "bb_head", "bb_maxat", "dyn"])
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sizes = [32 * k for k in range(20, 31)]  # 640..960
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "sizes": sizes, "variants": {}}
    for v in a.variants:
        try:
            res["variants"][v] = run(v, sizes, a.iters)
        except Exception as e:  # a failing variant (e.g. a compiler bug) is a result too
            res["variants"][v] = {"error": repr(e)[:400]}
            print(v, "FAILED", repr(e)[:300], flush=True)  # noqa: T201
        json.dump(res, open(a.out, "w"), indent=2)
    base = res["variants"].get("eager", {}).get("steady_mean_ms")
    for v, r in res["variants"].items():
        if "steady_mean_ms" in r and base:
            r["speedup"] = base / r["steady_mean_ms"]
            print(f"{v:9} steady {r['steady_mean_ms']:6.1f} ms  speed-up {r['speedup']:.3f}x  "  # noqa: T201
                  f"compile+autotune {r['compile_total_s'] / 60:5.1f} min")
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
