#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""DINOXHead + DFL + AdamW: full RTMDet-style head with distribution regression."""

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
        self.no_aug_epochs = 20
        self.basic_lr_per_img = 0.004 / 128.0
        self.adamw_lr = 0.004
        self.adamw_weight_decay = 0.05
        self.ema_momentum = 0.0002

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, DINOXHeadDFL

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = DINOXHeadDFL(
                self.num_classes, self.width, in_channels=in_channels, act=self.act,
                reg_max=16, dfl_weight=0.25,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            lr = self.warmup_lr if self.warmup_epochs > 0 else self.adamw_lr
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
