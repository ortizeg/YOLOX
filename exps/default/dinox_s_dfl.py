#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""DINO-X S + DFL experiment: DINOXHead with Distribution Focal Loss.

Builds on dinox_s (RTMDet-style assignment + QFL) and adds DFL for
distribution-based box regression with reg_max=16 (17 bins per edge).
"""

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, DINOXHead

        import torch.nn as nn

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
            head = DINOXHead(
                self.num_classes, self.width, in_channels=in_channels, act=self.act,
                use_dfl=True,
                reg_max=16,
                dfl_weight=0.25,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
