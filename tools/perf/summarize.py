#!/usr/bin/env python3
"""Turn a box_bench.sh output folder (check.json, step.json, data.json, e2e_*) into markdown tables.
  python tools/perf/summarize.py <perf_dir>  # noqa: T201 (prints the tables)"""

import json
import re
import statistics
import sys
from pathlib import Path


def e2e(d):
    out = {}
    for line in (d / "e2e_seconds.txt").read_text().split("\n"):
        if line.strip():
            k, v = line.split()
            out[k] = {"wall_s": int(v)}
    for k in out:
        log = (d / f"e2e_{k}_train_log.txt")
        if log.exists():
            t = [float(x) for x in re.findall(r"iter_time: ([0-9.]+)s", log.read_text())]
            dt = [float(x) for x in re.findall(r"data_time: ([0-9.]+)s", log.read_text())]
            ap = re.findall(r"Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all.*?= ([0-9.]+)", log.read_text())
            out[k].update(iter_s=statistics.median(t) if t else None, data_s=statistics.median(dt) if dt else None,
                          last_val_ap=float(ap[-1]) if ap else None)
    return out


def main():
    d = Path(sys.argv[1])
    print((d / "gpu.txt").read_text())  # noqa: T201
    c = json.loads((d / "check.json").read_text())
    print("\n### GPU equivalence (deterministic, 60 iterations, final EMA weights)\n")  # noqa: T201
    print("| flags | losses identical | max rel loss diff | weights identical | max abs weight diff |")  # noqa: T201
    print("|---|---|---:|---|---:|")  # noqa: T201
    for k, r in c["sets"].items():
        if k == "baseline":
            continue
        print(f"| {k} | {r.get('losses_identical')} | {r.get('loss_max_rel_diff', 0):.2e} | "  # noqa: T201
              f"{r.get('weights_identical')} | {r.get('weight_max_abs_diff', 0):.2e} |")
    s = json.loads((d / "step.json").read_text())
    print(f"\n### GPU step ({s['gpu']}, batch {s['batch']}, sizes {s['sizes'][0]}-{max(s['sizes'])})\n")  # noqa: T201
    print("| flags | ms/iter | img/s | speed-up |\n|---|---:|---:|---:|")  # noqa: T201
    for k, r in s["sets"].items():
        print(f"| {k} | {r['sec_per_iter_median'] * 1000:.1f} | {r['img_per_s']:.1f} | "  # noqa: T201
              f"{r['speedup_vs_baseline']:.2f}x |")
    da = json.loads((d / "data.json").read_text())
    print(f"\n### Data path ({da['cpu_count']} CPUs)\n\n| flags | workers | img/s |\n|---|---:|---:|")  # noqa: T201
    for r in da["rows"]:
        print(f"| {r['flags'][:40]} | {r['workers']} | {r['img_per_s']:.1f} |")  # noqa: T201
    e = e2e(d)
    print("\n### End to end, 20 epochs back to back\n\n| run | wall s | iter s | data s | last val AP |")  # noqa: T201
    print("|---|---:|---:|---:|---:|")  # noqa: T201
    for k, r in e.items():
        print(f"| {k} | {r['wall_s']} | {r.get('iter_s')} | {r.get('data_s')} | {r.get('last_val_ap')} |")  # noqa: T201
    if "base" in e and "perf" in e:
        print(f"\nend-to-end speed-up: {e['base']['wall_s'] / e['perf']['wall_s']:.2f}x wall clock")  # noqa: T201


if __name__ == "__main__":
    main()
