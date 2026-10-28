#!/usr/bin/env python3
"""Build the YOLOX basketball datasets (10-class and merged 5-class).

The 10-class jsons the published YOLOX-M run trained on were never copied off
the training box, so both taxonomies are rebuilt here from the Roboflow COCO
export (basketball-player-detection-3 v1). Images are symlinked, never copied.

10-class: drop the zero-annotation `basketball` category (id 0) and shift the
remaining ids down by one, so ids 0..9 match the published labels_mapping.json.
merged5: player <- player + 4 player-* actions; ball <- ball + ball-in-basket;
referee; rim; number (benchmarks/basketball/conf/taxonomy/merged5.yaml).

Every check below raises on failure; nothing is written for a split until the
checks for that split pass.

Usage:
    python scripts/build_basketball_datasets.py \
        --src ../../data/basketball-player-detection-3 --out-root datasets
"""

import argparse
import collections
import copy
import json
import os

import numpy as np

SPLITS = ("train", "valid", "test")

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

EXPECTED_ANNS = {"train": 9552, "valid": 1953, "test": 1980}
# docs/dataset.md, class_counts_merged5 table
EXPECTED_MERGED5 = {
    "player": (4556, 951, 944),
    "ball": (465, 97, 96),
    "referee": (1221, 287, 280),
    "rim": (455, 96, 93),
    "number": (2855, 522, 567),
}


def check(cond, msg):
    if not cond:
        raise SystemExit(f"ASSERTION FAILED: {msg}")
    print(f"  ok  {msg}")


def remap(src, class_names, name_of):
    """Return a copy of a COCO dict relabelled to class_names via name_of()."""
    old = {c["id"]: c["name"] for c in src["categories"]}
    new_id = {n: i for i, n in enumerate(class_names)}
    out = copy.deepcopy(src)
    out["categories"] = [
        {"id": i, "name": n, "supercategory": "basketball"} for i, n in enumerate(class_names)
    ]
    anns = []
    for a in out["annotations"]:
        name = old[a["category_id"]]
        if name == "basketball":
            raise SystemExit("ASSERTION FAILED: found an annotation on `basketball`")
        a["category_id"] = new_id[name_of(name)]
        anns.append(a)
    out["annotations"] = anns
    return out


def iou_matrix(b):
    """Pairwise IoU for xywh boxes."""
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = x1 + b[:, 2], y1 + b[:, 3]
    iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    inter = iw * ih
    area = b[:, 2] * b[:, 3]
    return inter / np.maximum(area[:, None] + area[None] - inter, 1e-9)


def high_iou_pairs(d, thr=0.9):
    """Count same-class box pairs (same image) with IoU > thr, per class name."""
    names = {c["id"]: c["name"] for c in d["categories"]}
    groups = collections.defaultdict(list)
    for a in d["annotations"]:
        groups[(a["image_id"], a["category_id"])].append(a["bbox"])
    counts = collections.Counter()
    for (_, cid), boxes in groups.items():
        if len(boxes) < 2:
            continue
        m = iou_matrix(np.asarray(boxes, dtype=np.float64))
        counts[names[cid]] += int((np.triu(m, k=1) > thr).sum())
    return counts


def link_images(src_split_dir, dst):
    if os.path.islink(dst):
        if os.path.realpath(dst) != os.path.realpath(src_split_dir):
            raise SystemExit(f"ASSERTION FAILED: {dst} links elsewhere")
        return
    os.symlink(os.path.realpath(src_split_dir), dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Roboflow COCO export root")
    p.add_argument("--out-root", default="datasets")
    args = p.parse_args()

    out10 = os.path.join(args.out_root, "basketball")
    out5 = os.path.join(args.out_root, "basketball_merged5")
    for root in (out10, out5):
        os.makedirs(os.path.join(root, "annotations"), exist_ok=True)

    totals = {}
    for i, split in enumerate(SPLITS):
        print(f"[{split}]")
        src_dir = os.path.join(args.src, split)
        with open(os.path.join(src_dir, "_annotations.coco.json")) as f:
            src = json.load(f)

        cats = sorted((c["id"], c["name"]) for c in src["categories"])
        check(cats == [(0, "basketball")] + list(enumerate(TEN_CLASS, start=1)),
              "source categories are basketball(0) + the 10 published classes in order")

        d10 = remap(src, TEN_CLASS, lambda n: n)
        d5 = remap(src, MERGED5, lambda n: MERGE.get(n, n))

        ids_src = [im["id"] for im in src["images"]]
        check([im["id"] for im in d10["images"]] == ids_src
              and [im["id"] for im in d5["images"]] == ids_src,
              f"identical image ids and order, 10c vs 5c ({len(ids_src)} images)")
        check(len(d10["annotations"]) == len(d5["annotations"]) == EXPECTED_ANNS[split],
              f"annotations conserved: {len(d5['annotations'])} == {EXPECTED_ANNS[split]}")
        check([a["id"] for a in d10["annotations"]] == [a["id"] for a in d5["annotations"]]
              and [a["bbox"] for a in d10["annotations"]] == [a["bbox"] for a in d5["annotations"]],
              "same annotation ids and boxes in 10c and 5c (only category_id differs)")

        c5 = collections.Counter(MERGED5[a["category_id"]] for a in d5["annotations"])
        for name, exp in EXPECTED_MERGED5.items():
            check(c5[name] == exp[i], f"merged5 {name}: {c5[name]} == {exp[i]}")
        check(set(c5) == set(MERGED5), "no classes outside merged5")

        pairs10 = high_iou_pairs(d10)
        pairs5 = high_iou_pairs(d5)
        print(f"  same-class pairs IoU>0.9: 10c {sum(pairs10.values())} {dict(pairs10)}"
              f" | 5c {sum(pairs5.values())} {dict(pairs5)}")
        totals[split] = (sum(pairs10.values()), sum(pairs5.values()))

        for root, d in ((out10, d10), (out5, d5)):
            with open(os.path.join(root, "annotations", f"{split}.json"), "w") as f:
                json.dump(d, f)
            link_images(src_dir, os.path.join(root, split))

    print("IoU>0.9 same-class pair totals (10c, 5c):", totals)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
