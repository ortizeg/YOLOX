#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOX-S + AdamW fine-tuning at 800x800 on basketball detection.

Same as yolox_s_basketball_adamw but trained at 800x800 instead of 640.
300 epochs with longer no-aug refinement.
"""

import os
import torch
import torch.nn as nn
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Train at 800x800 for higher resolution
        self.input_size = (800, 800)
        self.test_size = (800, 800)

        # Fine-tuning hyperparameters (AdamW)
        self.max_epoch = 300
        self.no_aug_epochs = 40
        self.warmup_epochs = 10
        self.eval_interval = 5

        # AdamW optimizer settings — lower LR for fine-tuning from COCO
        self.basic_lr_per_img = 0.0005 / 64.0
        self.adamw_lr = 0.0005
        self.adamw_weight_decay = 0.05

        # Enable mosaic to create synthetic variety from small dataset
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.5
        self.enable_mixup = True
        self.mosaic_scale = (0.5, 1.5)
        self.mixup_scale = (0.5, 1.5)

        # Augmentations
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.2
        self.shear = 2.0

        # Dataset paths
        self.data_dir = "datasets/basketball"
        self.train_ann = "train.json"
        self.val_ann = "valid.json"
        self.test_ann = "test.json"

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels, act=self.act,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act,
            )
            self.model = YOLOX(backbone, head)
            self.model.apply(init_yolo)
            self.model.head.initialize_biases(1e-2)

        self.model.train()
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=200,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name="valid",
            img_size=self.test_size,
            preproc=ValTransform(legacy=False),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            lr = self.warmup_lr if self.warmup_epochs > 0 else self.adamw_lr

            pg0, pg1, pg2 = [], [], []
            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    if v.bias.requires_grad:
                        pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    if hasattr(v, "weight") and v.weight is not None and v.weight.requires_grad:
                        pg0.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    if v.weight.requires_grad:
                        pg1.append(v.weight)

            optimizer = torch.optim.AdamW(pg0, lr=lr, weight_decay=0.0)
            optimizer.add_param_group({"params": pg1, "weight_decay": self.adamw_weight_decay})
            optimizer.add_param_group({"params": pg2, "weight_decay": 0.0})
            self.optimizer = optimizer

        return self.optimizer
