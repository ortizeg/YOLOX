#!/usr/bin/env python3
"""Evaluation speed and agreement: evaluate one checkpoint three times in a row with the original evaluator and
with the eval flags (persistent workers, fp16 autocast). Reports AP50:95 / AP50 and seconds per evaluation.
  python tools/perf/eval_check.py -f <exp> -c best_ckpt.pth --out eval_check.json -- [exp opts]
"""

import argparse
import json
import time

import torch

from yolox.exp import get_exp
from yolox.utils import perf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--exp", required=True)
    p.add_argument("-c", "--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    opts = a.opts[1:] if a.opts[:1] == ["--"] else a.opts
    res = {}
    for label, spec in [("original", ""), ("eval_persistent", "eval_persistent"),
                        ("eval_persistent+eval_amp", "eval_persistent,eval_amp")]:
        perf.set_flags(spec)
        exp = get_exp(a.exp, None)
        exp.merge(opts)
        model = exp.get_model()
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"])
        model.cuda().eval()
        evaluator = exp.get_evaluator(batch_size=16, is_distributed=False)
        runs = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            (ap, ap50, _), _ = exp.eval(model, evaluator, False, return_outputs=True)
            torch.cuda.synchronize()
            runs.append({"seconds": round(time.perf_counter() - t0, 3), "ap50_95": ap, "ap50": ap50})
        res[label] = runs
        print(label, runs)  # noqa: T201
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
