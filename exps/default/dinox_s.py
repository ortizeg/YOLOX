#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""DINO-X S experiment: YOLOX-S with RTMDet-style label assignment and QFL.

Uses the same backbone (CSPDarknet) and neck (PAFPN) as YOLOX-S but replaces
the detection head with DINOXHead, which adds:
- Soft classification cost in SimOTA assignment
- QualityFocalLoss for classification
- Soft center prior (exponential decay)
- IoU-weighted regression loss
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
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
