#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOX-S + AdamW experiment: RTMDet training hyperparameters on standard YOLOX.

Tests whether mmyolo's RTMDet training hyperparameters improve standard
YOLOX-S (with no head/loss changes). This is a training recipe comparison,
not an architecture comparison.

Changes from YOLOX-S default:
- AdamW optimizer (lr=0.004, weight_decay=0.05) replacing SGD
- no_aug_epochs: 20 (from 15)
- EMA momentum: 0.0002 (from 0.0001)

Reference: https://github.com/open-mmlab/mmyolo/tree/main/configs/yolox
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

        # RTMDet training hyperparameters (from mmyolo)
        self.no_aug_epochs = 20  # was 15
        # AdamW: fixed LR=0.004, not scaled by batch size
        # basic_lr_per_img * batch_size = 0.004, so for batch 128:
        self.basic_lr_per_img = 0.004 / 128.0
        self.adamw_lr = 0.004
        self.adamw_weight_decay = 0.05
        self.ema_momentum = 0.0002  # was 0.0001

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.adamw_lr

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases — no weight decay
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # norms — no weight decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # regular weights — with decay

            optimizer = torch.optim.AdamW(
                pg0, lr=lr, weight_decay=0.0,  # no decay for norms
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.adamw_weight_decay}
            )
            optimizer.add_param_group(
                {"params": pg2, "weight_decay": 0.0}  # no decay for biases
            )
            self.optimizer = optimizer

        return self.optimizer
