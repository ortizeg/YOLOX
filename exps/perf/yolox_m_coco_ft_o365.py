#!/usr/bin/env python3
"""YOLOX-M COCO fine-tune from the Objects365 pretrain (yolox_m_objects365_pretrain.py best_ckpt.pth, -c).

Everything but the 365-way cls_preds loads (load_ckpt skips shape mismatches; the 80-way classifier is new).
Recipe = YOLOX's YOLOX-M COCO recipe (SGD 0.01/64 per image, mosaic + mixup, multiscale, EMA) on a shorter
schedule: 80 epochs, warm-up 3, no-aug last 10, eval every 5 on val2017 (the official 46.9 is 300 epochs from scratch)."""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = "datasets/COCO"
        self.max_epoch = 80
        self.warmup_epochs = 3
        self.no_aug_epochs = 10
        self.eval_interval = 5
        self.print_interval = 50
        self.data_num_workers = 8
        self.eval_num_workers = 4
        self.save_history_ckpt = False
