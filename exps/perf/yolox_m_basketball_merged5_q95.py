#!/usr/bin/env python3
"""Accuracy check for a pre-resized JPEG training cache (perf docs, tier C).

Identical to exps/default/yolox_m_basketball_merged5_adamw.py, except that the TRAIN images are replaced by
copies resized exactly as read_img() would (long side fits input_size, cv2 INTER_LINEAR) and re-encoded as JPEG
quality JPEG_Q. Annotations are untouched: load_anno scales boxes from the json's original width/height, and
read_img's resize becomes a no-op on the smaller file, so the only change is JPEG re-quantisation. Val/test
images are the originals. The copy is built once, in the main process, before the dataloader starts.
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "default"))
from yolox_m_basketball_merged5_adamw import Exp as Base  # noqa: E402

JPEG_Q = 95


def build_copy(src_dir, dst_dir, size, q, file_names):
    """Re-encode exactly the images the annotation file lists (skips stray files such as macOS ._ metadata)."""
    done = os.path.join(dst_dir, ".complete")
    if os.path.exists(done):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for f in file_names:
        img = cv2.imread(os.path.join(src_dir, f))
        assert img is not None, f"unreadable training image {f}"
        r = min(size[0] / img.shape[0], size[1] / img.shape[1])
        img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
        os.makedirs(os.path.dirname(os.path.join(dst_dir, f)) or dst_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dst_dir, f), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    open(done, "w").write(f"q={q} size={size} images={len(file_names)}\n")


class Exp(Base):
    def __init__(self):
        super().__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        name = f"train_q{JPEG_Q}_{self.input_size[0]}x{self.input_size[1]}"
        import json
        with open(os.path.join(self.data_dir, "annotations", self.train_ann)) as fh:
            files = [im["file_name"] for im in json.load(fh)["images"]]
        build_copy(os.path.join(self.data_dir, "train"), os.path.join(self.data_dir, name), self.input_size, JPEG_Q,
                   files)
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir, json_file=self.train_ann, name=name, img_size=self.input_size,
            preproc=TrainTransform(max_labels=200, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache, cache_type=cache_type)
