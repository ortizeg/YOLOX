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


class Integral(nn.Module):
    """Convert discrete distribution logits to a point estimate.

    Computes ``sum(softmax(logits) * [0, 1, ..., reg_max])``.
    Reference: GFL paper (arXiv 2006.04388).
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x).unsqueeze(0))
        if len(shape) >= 2 and shape[-1] == 4 * (self.reg_max + 1):
            x = x.reshape(*shape[:-1], 4)
        return x


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss from GFL paper (arXiv 2006.04388).

    For a continuous target ``y`` between bins ``y_i`` and ``y_{i+1}``::

        DFL = (y_{i+1} - y) * CE(logits, y_i) + (y - y_i) * CE(logits, y_{i+1})
    """

    def __init__(self, reduction: str = "none") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_bins = pred.shape[-1]
        target = target.clamp(min=0, max=n_bins - 1 - 0.01)
        dis_left = target.long()
        dis_right = (dis_left + 1).clamp(max=n_bins - 1)
        weight_left = dis_right.float() - target
        weight_right = target - dis_left.float()
        loss = (
            F.cross_entropy(pred, dis_left, reduction="none") * weight_left
            + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
