#!/usr/bin/env python3
"""YOLOX-M on Objects365 (365 classes), 640 input: YOLOX's own YOLOX-M recipe (exps/default/yolox_m.py:
depth 0.67, width 0.75, SGD, 300-epoch schedule, mosaic + mixup) pointed at an Objects365 COCO-format json.
Used for the 8-GPU throughput test; data_dir / train_ann / val_ann are set by opts or the box script."""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.67
        self.width = 0.75
        self.num_classes = 365
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = "datasets/objects365"
        self.train_ann = "train_subset.json"
        self.val_ann = "train_subset_val.json"
        self.eval_interval = 10_000  # throughput runs: no evaluation

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(data_dir=self.data_dir, json_file=self.train_ann, name="train", img_size=self.input_size,
                           preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                           cache=cache, cache_type=cache_type)

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(data_dir=self.data_dir, json_file=self.val_ann, name="train", img_size=self.test_size,
                           preproc=ValTransform(legacy=False))
