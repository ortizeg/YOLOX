#!/usr/bin/env python3
"""CPU samples/s/core of the full training sample (mosaic + mixup + TrainTransform) on synthetic Objects365-like data
(640 px JPEGs, ~20 boxes, no image cache), with and without decode_reuse."""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from yolox.data import COCODataset, MosaicDetection, TrainTransform
from yolox.utils import perf


def make_data(root: Path, n: int):
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "train").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    images, anns, aid = [], [], 1
    for i in range(n):
        h, w = (480, 640) if i % 3 else (640, 480)
        base = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
        img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imwrite(str(root / "train" / f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        images.append({"id": i + 1, "file_name": f"{i:06d}.jpg", "height": h, "width": w})
        for _ in range(20):
            bw, bh = rng.uniform(10, w / 3), rng.uniform(10, h / 3)
            x, y = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
            anns.append({"id": aid, "image_id": i + 1, "category_id": int(rng.integers(1, 6)),
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1
    cats = [{"id": c, "name": str(c)} for c in range(1, 6)]
    (root / "annotations" / "train.json").write_text(json.dumps({"images": images, "annotations": anns,
                                                                "categories": cats}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--base", default="uint8_transport,skip_noop_resize,hsv_lut,int_mixup,cv_flip,preproc_nopad,"
                                     "hwc_transport,fast_anno,class_map")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    root = Path(a.root)
    if not (root / "annotations" / "train.json").exists():
        make_data(root, a.n)
    res = {}
    for name, extra, prob in [("base", "", 1.0), ("reuse_p0.5", ",decode_reuse", 0.5), ("reuse_p1", ",decode_reuse", 1.0)]:
        perf.set_flags(a.base + extra)
        ds = COCODataset(data_dir=str(root), json_file="train.json", name="train", img_size=(a.size, a.size))
        m = MosaicDetection(ds, img_size=(a.size, a.size), preproc=TrainTransform(max_labels=120, flip_prob=0.5,
                            hsv_prob=1.0), degrees=10.0, translate=0.1, mosaic_scale=(0.1, 2.0),
                            mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True, mosaic_prob=1.0, mixup_prob=0.5,
                            reuse_pool=64, reuse_prob=prob)
        random.seed(0)
        np.random.seed(0)
        order = list(range(len(m)))
        random.shuffle(order)
        for i in order[:50]:  # warm the pool and the page cache
            m[i]
        t = time.perf_counter()
        for k in range(a.samples):
            m[order[(50 + k) % len(order)]]
        dt = time.perf_counter() - t
        res[name] = a.samples / dt
        logger.info(f"{name}: {res[name]:.1f} samples/s/core")
    for k in res:
        logger.info(f"{k}: {res[k] / res['base']:.2f}x")
    Path(a.out).write_text(json.dumps({"size": a.size, "samples_per_s_core": res}, indent=2))


if __name__ == "__main__":
    main()
