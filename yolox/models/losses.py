#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class QualityFocalLoss(nn.Module):
    """Quality Focal Loss from Generalized Focal Loss paper (arXiv 2006.04388).

    Used by RTMDet for classification with soft IoU targets. For positives,
    the target is the IoU quality score (not binary 1.0), and the modulating
    factor ``|target - sigmoid(pred)|^beta`` down-weights well-calibrated
    predictions while amplifying poorly-calibrated ones.

    For negatives (target == 0), this reduces to standard focal loss:
    ``BCE(pred, 0) * sigmoid(pred)^beta``.

    Args:
        beta: Exponent for the modulating factor. Default ``2.0`` (RTMDet).
        reduction: ``"none"`` | ``"mean"`` | ``"sum"``.
    """

    def __init__(self, beta: float = 2.0, reduction: str = "none") -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute QFL.

        Args:
            pred: Classification logits of shape ``(N, C)``.
            target: Soft targets of shape ``(N, C)`` where positive entries
                contain IoU quality scores and negatives are ``0.0``.

        Returns:
            Per-element loss of shape ``(N, C)`` when ``reduction="none"``.
        """
        pred_sigmoid = pred.sigmoid()
        scale_factor = (pred_sigmoid - target).abs().pow(self.beta)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none",
        ) * scale_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
