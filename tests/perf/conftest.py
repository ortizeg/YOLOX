"""Shared fixtures for the perf equivalence tests: a small synthetic COCO dataset and a flag switcher."""

import json
import random
from contextlib import contextmanager

import cv2
import numpy as np
import pytest

from yolox.utils import perf


@contextmanager
def flags(spec):
    old = perf.flags()
    perf.set_flags(spec)
    try:
        yield
    finally:
        perf.set_flags(old)


def seed_all(s):
    random.seed(s)
    np.random.seed(s)


@pytest.fixture(scope="session")
def coco_dir(tmp_path_factory):
    """16 images (mixed sizes, textured so HSV/resize/warp matter), non-contiguous category ids, crowded and
    tiny boxes (tiny-only images exercise TrainTransform's fallback path)."""
    root = tmp_path_factory.mktemp("coco")
    (root / "annotations").mkdir()
    (root / "train").mkdir()
    rng = np.random.default_rng(0)
    sizes = [(720, 1280), (480, 640), (900, 900), (1080, 1920), (600, 800), (360, 640), (1024, 768), (512, 512)]
    cats = [1, 3, 7, 12, 40]
    images, anns = [], []
    aid = 1
    for i in range(16):
        h, w = sizes[i % len(sizes)]
        base = rng.integers(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
        img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        img = cv2.add(img, rng.integers(0, 40, (h, w, 3), dtype=np.uint8))
        cv2.imwrite(str(root / "train" / f"{i:04d}.jpg"), img)
        images.append({"id": i + 1, "file_name": f"{i:04d}.jpg", "height": h, "width": w})
        tiny_only = i == 5
        n = 2 if tiny_only else int(rng.integers(1, 25))
        for _ in range(n):
            if tiny_only:
                bw, bh = 1.2, 1.0
            else:
                bw, bh = float(rng.uniform(4, w / 3)), float(rng.uniform(4, h / 3))
            x, y = float(rng.uniform(0, w - bw)), float(rng.uniform(0, h - bh))
            anns.append({"id": aid, "image_id": i + 1, "category_id": int(rng.choice(cats)),
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1
    js = {"images": images, "annotations": anns,
          "categories": [{"id": c, "name": f"c{c}"} for c in cats]}
    (root / "annotations" / "train.json").write_text(json.dumps(js))
    return root
