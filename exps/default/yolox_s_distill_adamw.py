#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""YOLOX-S + DINOv2 distillation + AdamW.

Standard YOLOX-S with frozen DINOv2-B/14 teacher for feature distillation.
Uses interpolated loss (Hinton et al.): L = α*L_det + (1-α)*L_distill.
Teacher is removed at inference — zero overhead.
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

        # Distillation config
        self.distill_alpha = 0.3  # 30% detection, 70% distillation (teacher-focused)
        self.distill_levels = [4, 8, 12]
        self.teacher_model = "dinov2_vitb14"

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        from yolox.models.distill import YOLOXDistill

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
            base_model = YOLOX(backbone, head)
            base_model.apply(init_yolo)
            base_model.head.initialize_biases(1e-2)

            # Wrap with distillation
            student_channels = [int(c * self.width) for c in in_channels]
            self.model = YOLOXDistill(
                model=base_model,
                teacher_model=self.teacher_model,
                distill_alpha=self.distill_alpha,
                distill_levels=self.distill_levels,
                student_channels=student_channels,
            )

        self.model.train()
        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            lr = self.warmup_lr if self.warmup_epochs > 0 else self.adamw_lr

            pg0, pg1, pg2 = [], [], []
            for k, v in self.model.named_modules():
                # Skip frozen teacher parameters
                if "teacher" in k:
                    continue
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    if v.bias.requires_grad:
                        pg2.append(v.bias)
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    if v.weight.requires_grad:
                        pg0.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    if v.weight.requires_grad:
                        pg1.append(v.weight)

            optimizer = torch.optim.AdamW(pg0, lr=lr, weight_decay=0.0)
            optimizer.add_param_group({"params": pg1, "weight_decay": self.adamw_weight_decay})
            optimizer.add_param_group({"params": pg2, "weight_decay": 0.0})
            self.optimizer = optimizer

        return self.optimizer
