#!/usr/bin/env python3
"""YOLOX-S pretraining on Objects365 v2 from scratch (365 classes, 640 input), for an Objects365 -> COCO fine-tune.

Recipe = YOLOX's own YOLOX-M COCO recipe (SGD 0.01/64 per image, wd 5e-4, mosaic + mixup, multiscale 640 +- 5x32,
EMA) with the schedule rescaled to Objects365: 20 epochs x 1.74 M images ~= the 300 x 118 k image passes of the COCO
recipe. Warm-up 1 epoch, no-aug last 2 epochs. Val = 5 000 images held out of train (o365_prep.sh), every 2 epochs.
max_labels 300 (crowded mosaics: 19.5 boxes per image, up to ~4x + mixup per sample)."""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 365
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = "datasets/objects365"
        self.train_ann = "o365_train.json"
        self.val_ann = "o365_val5k.json"
        self.max_epoch = 20
        self.warmup_epochs = 1
        self.no_aug_epochs = 2
        self.eval_interval = 2
        self.print_interval = 50
        self.max_labels = 300
        self.data_num_workers = 8
        self.eval_num_workers = 4  # eval workers stay alive beside the training ones (eval_persistent); keep threads in check
        self.save_history_ckpt = False  # latest + best only; latest is synced off the box every few minutes

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(data_dir=self.data_dir, json_file=self.train_ann, name="train", img_size=self.input_size,
                           preproc=TrainTransform(max_labels=self.max_labels, flip_prob=self.flip_prob,
                                                  hsv_prob=self.hsv_prob),
                           cache=cache, cache_type=cache_type)

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(data_dir=self.data_dir, json_file=self.val_ann, name="train", img_size=self.test_size,
                           preproc=ValTransform(legacy=False))
