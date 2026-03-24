#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOX-S + MAL + AdamW: Matchability-Aware Loss with RTMDet training recipe.

Standard YOLOX-S architecture with:
- Matchability-Aware Loss (DEIM) for classification reweighting
- AdamW optimizer (lr=0.004, wd=0.05)
- no_aug_epochs=20, ema_momentum=0.0002
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
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # RTMDet training hyperparameters
        self.no_aug_epochs = 20
        self.basic_lr_per_img = 0.004 / 128.0
        self.adamw_lr = 0.004
        self.adamw_weight_decay = 0.05
        self.ema_momentum = 0.0002

        # MAL config — gamma=0.5 for YOLOX (geometric mean of IoU and cls)
        # DEIM uses 1.5 for one-to-one detectors, but gamma>1 causes
        # cls_score^(1-gamma) to explode for small cls_scores in YOLOX's
        # one-to-many framework, clamping all matchability targets to 1.0
        self.mal_gamma = 0.75

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHeadMAL

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
            head = YOLOXHeadMAL(
                self.num_classes, self.width, in_channels=in_channels, act=self.act,
                mal_gamma=self.mal_gamma,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.adamw_lr

            pg0, pg1, pg2 = [], [], []
            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)

            optimizer = torch.optim.AdamW(pg0, lr=lr, weight_decay=0.0)
            optimizer.add_param_group({"params": pg1, "weight_decay": self.adamw_weight_decay})
            optimizer.add_param_group({"params": pg2, "weight_decay": 0.0})
            self.optimizer = optimizer

        return self.optimizer
