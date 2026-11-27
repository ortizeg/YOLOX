#!/usr/bin/env python3
"""Compare image-cache formats for the training data path, single core, on a folder of JPEGs.

For each format: one-time build cost per image, bytes per image, and per-read cost of what the dataset's
read_img() needs (the image resized so its long side fits img_size, uint8 BGR).
  python tools/perf/bench_cache.py <jpeg_dir> <work_dir> --size 640 --n 200
Formats:
  jpeg_full   : no cache, decode the original + resize (today without --cache)
  jpeg_reduced: no cache, libjpeg DCT-domain downscale (IMREAD_REDUCED_COLOR_2/4) + resize   [pixels differ]
  npy         : YOLOX --cache disk today: one .npy per image, raw uint8 at img_size              [identical]
  shard       : raw uint8 at img_size packed into one memmap file + offset index                 [identical]
  jpeg_q95    : re-encoded JPEG at img_size, quality 95                                          [pixels differ]
  png         : lossless PNG at img_size (compression 1)                                         [identical]
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)


def resized(img, size):
    r = min(size / img.shape[0], size / img.shape[1])
    return cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)


def reduced_read(path, size):
    h, w = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8).shape[:2]  # cheap header-ish probe
    long_full = max(h, w) * 8
    flag = cv2.IMREAD_COLOR
    if long_full >= 4 * size:
        flag = cv2.IMREAD_REDUCED_COLOR_4
    elif long_full >= 2 * size:
        flag = cv2.IMREAD_REDUCED_COLOR_2
    return resized(cv2.imread(str(path), flag), size)


def timeit(fn, items):
    t0 = time.perf_counter()
    for it in items:
        fn(it)
    return (time.perf_counter() - t0) / len(items) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("work")
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    files = sorted(Path(a.src).glob("*.jpg"))[: a.n]
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    S = a.size
    ref = {f: resized(cv2.imread(str(f)), S) for f in files}
    res = {"n": len(files), "size": S, "src_bytes_per_img": int(np.mean([f.stat().st_size for f in files])),
           "src_shape": list(cv2.imread(str(files[0])).shape), "formats": {}}

    # builds
    t0 = time.perf_counter()
    for f in files:
        np.save(work / (f.stem + ".npy"), resized(cv2.imread(str(f)), S))
    b_npy = (time.perf_counter() - t0) / len(files) * 1000
    t0 = time.perf_counter()
    for f in files:
        cv2.imwrite(str(work / (f.stem + ".jpg")), resized(cv2.imread(str(f)), S), [cv2.IMWRITE_JPEG_QUALITY, 95])
    b_jpg = (time.perf_counter() - t0) / len(files) * 1000
    t0 = time.perf_counter()
    for f in files:
        cv2.imwrite(str(work / (f.stem + ".png")), resized(cv2.imread(str(f)), S), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    b_png = (time.perf_counter() - t0) / len(files) * 1000
    t0 = time.perf_counter()
    arrs = [resized(cv2.imread(str(f)), S) for f in files]
    offs, shapes, o = [], [], 0
    for x in arrs:
        offs.append(o); shapes.append(x.shape); o += x.nbytes
    mm = np.lib.format.open_memmap(work / "shard.npy", mode="w+", dtype=np.uint8, shape=(o,))
    for x, of in zip(arrs, offs):
        mm[of:of + x.nbytes] = x.reshape(-1)
    mm.flush()
    del mm
    b_shard = (time.perf_counter() - t0) / len(files) * 1000
    shard = np.load(work / "shard.npy", mmap_mode="r")

    def rd_shard(i):
        of, sh = offs[i], shapes[i]
        return np.array(shard[of:of + int(np.prod(sh))]).reshape(sh)

    idx = list(range(len(files)))
    reads = {
        "jpeg_full": (lambda f: resized(cv2.imread(str(f)), S), files, None, 0),
        "jpeg_reduced": (lambda f: reduced_read(f, S), files, None, 0),
        "npy": (lambda f: np.load(work / (f.stem + ".npy")), files, b_npy, "npy"),
        "shard": (rd_shard, idx, b_shard, "shard"),
        "jpeg_q95": (lambda f: cv2.imread(str(work / (f.stem + ".jpg"))), files, b_jpg, "jpg"),
        "png": (lambda f: cv2.imread(str(work / (f.stem + ".png"))), files, b_png, "png"),
    }
    for name, (fn, items, build_ms, ext) in reads.items():
        fn(items[0])
        ms = min(timeit(fn, items) for _ in range(2))  # second pass: page cache warm, as in training
        out0 = fn(items[0])
        same = bool(np.array_equal(out0, ref[files[0]]))
        if ext == "shard":
            nbytes = os.path.getsize(work / "shard.npy") / len(files)
        elif ext:
            nbytes = np.mean([os.path.getsize(work / (f.stem + "." + ext)) for f in files])
        else:
            nbytes = res["src_bytes_per_img"]
        res["formats"][name] = {"read_ms": round(ms, 2), "build_ms": None if build_ms is None else round(build_ms, 2),
                                "bytes_per_img": int(nbytes), "identical_to_jpeg_full": same}
        print(f"{name:<13} read {ms:6.2f} ms  build {build_ms or 0:6.1f} ms  {nbytes / 1024:8.1f} KiB  identical={same}")  # noqa: T201
    json.dump(res, open(work / "bench_cache.json", "w"), indent=2)


if __name__ == "__main__":
    main()
