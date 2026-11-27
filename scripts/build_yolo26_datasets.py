#!/usr/bin/env python3
"""Build YOLO-format 10-class and merged 5-class basketball datasets for YOLO26.

Uses yolo26-basketball-training/scripts/convert_coco_to_yolo.py unchanged, so the
10-class arm is converted exactly like the published YOLO26m (name-based remap
onto the canonical 10-class order). The 5-class arm uses the same converter with
a name -> merged5 remap. Only train and valid are built; test is never needed for
training.

Checks: per-split image counts, annotation conservation, merged5 class counts,
and that every 5-class label file is the 10-class file with only the class id
remapped (same boxes, same order).

Usage:
    python scripts/build_yolo26_datasets.py --src /workspace/data \
        --converter-dir /workspace/yolo26/scripts --out-root /workspace/yolo26_data
"""

import argparse
import collections
import json
import os
import sys

import yaml

TEN_CLASS = [
    "ball", "ball-in-basket", "number", "player", "player-in-possession",
    "player-jump-shot", "player-layup-dunk", "player-shot-block", "referee", "rim",
]
MERGED5 = ["player", "ball", "referee", "rim", "number"]
MERGE = {
    "player-in-possession": "player",
    "player-jump-shot": "player",
    "player-layup-dunk": "player",
    "player-shot-block": "player",
    "ball-in-basket": "ball",
}
EXPECTED = {
    "train": {"images": 464, "anns": 9552, "merged5": (4556, 465, 1221, 455, 2855)},
    "valid": {"images": 96, "anns": 1953, "merged5": (951, 97, 287, 96, 522)},
}


def check(cond, msg):
    if not cond:
        raise SystemExit(f"ASSERTION FAILED: {msg}")
    print(f"  ok  {msg}")


def read_labels(split_dir):
    out = {}
    for fn in sorted(os.listdir(os.path.join(split_dir, "labels"))):
        with open(os.path.join(split_dir, "labels", fn)) as f:
            out[fn] = [line.split() for line in f.read().splitlines() if line.strip()]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="dir with {train,valid}/ images + _annotations.coco.json")
    p.add_argument("--converter-dir", required=True)
    p.add_argument("--out-root", required=True)
    args = p.parse_args()

    sys.path.insert(0, args.converter_dir)
    from convert_coco_to_yolo import build_category_remap, convert_split

    from pathlib import Path

    src_train_json = Path(args.src) / "train" / "_annotations.coco.json"
    remap10 = build_category_remap(src_train_json)
    with open(src_train_json) as f:
        cats = {c["id"]: c["name"] for c in json.load(f)["categories"]}
    remap5 = {cid: MERGED5.index(MERGE.get(n, n)) for cid, n in cats.items() if n in TEN_CLASS}
    check(len(remap10) == 10 and len(remap5) == 10, "10 annotated categories remapped in both arms")

    for arm, remap, names in (("10c", remap10, TEN_CLASS), ("5c", remap5, MERGED5)):
        root = Path(args.out_root) / arm
        for split in ("train", "valid"):
            convert_split(Path(args.src) / split, root / split, remap)
        with open(root / "data.yaml", "w") as f:
            yaml.dump({"path": str(root), "train": "train/images", "val": "valid/images",
                       "test": "", "names": dict(enumerate(names))}, f, default_flow_style=False)

    for split, exp in EXPECTED.items():
        print(f"[{split}]")
        l10 = read_labels(Path(args.out_root) / "10c" / split)
        l5 = read_labels(Path(args.out_root) / "5c" / split)
        check(len(l10) == len(l5) == exp["images"], f"{len(l10)} label files per arm")
        check(sorted(l10) == sorted(l5), "same label file names in both arms")
        n = sum(len(v) for v in l10.values())
        check(n == sum(len(v) for v in l5.values()) == exp["anns"], f"annotations conserved: {n}")
        to5 = [MERGED5.index(MERGE.get(c, c)) for c in TEN_CLASS]
        same = all(
            [str(to5[int(a[0])])] + a[1:] == b
            for fn in l10 for a, b in zip(l10[fn], l5[fn], strict=True)
        )
        check(same, "5c labels == 10c labels with only class id remapped")
        cnt = collections.Counter(MERGED5[int(r[0])] for v in l5.values() for r in v)
        check(tuple(cnt[c] for c in MERGED5) == exp["merged5"],
              f"merged5 counts {tuple(cnt[c] for c in MERGED5)}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
