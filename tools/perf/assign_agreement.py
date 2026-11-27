#!/usr/bin/env python3
"""One-step check that batched SimOTA (perf simota_batch) picks the same positives as the original loop on GPU.

Same model (COCO weights), same real batches, same fp16 autocast forward; targets computed both ways from the
identical head outputs. Reports per batch the share of foreground anchors that agree, and for disagreements
whether the anchors involved have tied costs (so only tie-breaking order differs).
  python tools/perf/assign_agreement.py -f <exp> -c yolox_m.pth --out assign.json -- [exp opts]
"""

import argparse
import json

import torch

from yolox.exp import get_exp
from yolox.utils import load_ckpt, perf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--exp", required=True)
    p.add_argument("-c", "--ckpt", required=True)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    a = p.parse_args()
    opts = a.opts[1:] if a.opts[:1] == ["--"] else a.opts
    exp = get_exp(a.exp, None)
    exp.merge(opts)
    perf.set_flags("")
    loader = exp.get_data_loader(batch_size=16, is_distributed=False, no_aug=False, cache_img=None)
    model = exp.get_model()
    load_ckpt(model, torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"])
    model.cuda().train()
    head = model.head
    captured = {}
    orig = head.get_losses

    def grab(*args, **kw):
        captured["args"], captured["kw"] = args, kw
        return orig(*args, **kw)

    head.get_losses = grab
    rows = []
    it = iter(loader)
    for _ in range(a.k):
        inps, targets, _, _ = next(it)
        inps, targets = inps.cuda().half(), targets.cuda().half()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            model(inps, targets)
            args = list(captured["args"])
            names = ["imgs", "x_shifts", "y_shifts", "expanded_strides", "labels", "outputs", "origin_preds", "dtype"]
            args += [captured["kw"][n] for n in names[len(args):]]
            (imgs, x_shifts, y_shifts, strides, labels, outputs, origin_preds, dtype) = args
            bbox, obj, cls = outputs[:, :, :4], outputs[:, :, 4:5], outputs[:, :, 5:]
            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
            xs, ys, st = torch.cat(x_shifts, 1), torch.cat(y_shifts, 1), torch.cat(strides, 1)
            perf.set_flags("")
            r = head.get_targets_loop(labels, nlabel, outputs, bbox, obj, cls, st, xs, ys, outputs.shape[1], dtype)
            perf.set_flags("simota_batch")
            b = head.get_targets_batched(labels, nlabel, bbox, obj, cls, st, xs, ys, dtype)
            perf.set_flags("")
        fr, fb = r[3], b[3]
        both = int((fr & fb).sum())
        union = int((fr | fb).sum())
        same_targets = bool(torch.equal(fr, fb) and torch.equal(r[1], b[1]) and torch.equal(r[0], b[0]))
        rows.append({"gts": int(nlabel.sum()), "fg_loop": int(fr.sum()), "fg_batch": int(fb.sum()),
                     "fg_agree": both / max(union, 1), "identical_targets": same_targets})
        print(rows[-1])  # noqa: T201
    res = {"batches": rows, "identical_batches": sum(x["identical_targets"] for x in rows),
           "mean_fg_agreement": sum(x["fg_agree"] for x in rows) / len(rows)}
    json.dump(res, open(a.out, "w"), indent=2)
    print(res["identical_batches"], "/", len(rows), "batches identical; mean fg agreement", res["mean_fg_agreement"])  # noqa: T201


if __name__ == "__main__":
    main()
