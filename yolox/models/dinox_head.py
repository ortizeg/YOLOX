#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# DINO-X Head: YOLOX head with RTMDet-style soft label assignment and QFL.
# Phase 2 adds Distribution Focal Loss (DFL) for box regression.

from __future__ import annotations

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid

from .losses import DistributionFocalLoss, IOUloss, Integral, QualityFocalLoss
from .network_blocks import BaseConv, DWConv


def _distance2bbox(
    points: torch.Tensor,
    distance: torch.Tensor,
    stride: torch.Tensor,
) -> torch.Tensor:
    """Decode (l, t, r, b) distances from anchor points to xyxy boxes.

    Args:
        points: Anchor center coordinates ``(N, 2)`` in absolute pixels.
        distance: Predicted distances ``(N, 4)`` as (l, t, r, b) in stride units.
        stride: Per-anchor stride ``(N,)`` or ``(N, 1)``.

    Returns:
        Boxes in cxcywh format ``(N, 4)`` for compatibility with existing IoU loss.
    """
    stride = stride.reshape(-1, 1) if stride.dim() == 1 else stride
    x1 = points[:, 0:1] - distance[:, 0:1] * stride
    y1 = points[:, 1:2] - distance[:, 1:2] * stride
    x2 = points[:, 0:1] + distance[:, 2:3] * stride
    y2 = points[:, 1:2] + distance[:, 3:4] * stride
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.cat([cx, cy, w, h], dim=-1)


def _bbox2distance(
    points: torch.Tensor,
    bbox: torch.Tensor,
    stride: torch.Tensor,
    reg_max: float,
) -> torch.Tensor:
    """Encode cxcywh boxes to (l, t, r, b) distances from anchor points.

    Args:
        points: Anchor center coordinates ``(N, 2)`` in absolute pixels.
        bbox: GT boxes ``(N, 4)`` in cxcywh format.
        stride: Per-anchor stride ``(N,)`` or ``(N, 1)``.
        reg_max: Maximum distance in stride units (for clamping).

    Returns:
        Distances ``(N, 4)`` as (l, t, r, b) in stride units, clamped to ``[0, reg_max]``.
    """
    stride = stride.reshape(-1, 1) if stride.dim() == 1 else stride
    # Convert cxcywh to xyxy
    x1 = bbox[:, 0:1] - bbox[:, 2:3] / 2
    y1 = bbox[:, 1:2] - bbox[:, 3:4] / 2
    x2 = bbox[:, 0:1] + bbox[:, 2:3] / 2
    y2 = bbox[:, 1:2] + bbox[:, 3:4] / 2
    # Distances from anchor center, normalized by stride
    left = (points[:, 0:1] - x1) / stride
    top = (points[:, 1:2] - y1) / stride
    right = (x2 - points[:, 0:1]) / stride
    bottom = (y2 - points[:, 1:2]) / stride
    return torch.cat([left, top, right, bottom], dim=-1).clamp(min=0, max=reg_max - 0.01)


class DINOXHead(nn.Module):
    """YOLOX detection head with RTMDet-style improvements.

    Differences from YOLOXHead:
    1. Soft classification cost in assignment: BCE(logits, Y_soft) * |Y_soft - P|^2
    2. QualityFocalLoss for training cls loss (replaces plain BCE on soft targets)
    3. Soft center prior: exponential decay instead of hard binary mask
    4. IoU-weighted regression loss
    5. (Phase 2) Optional DFL: distribution-based box regression with Integral decoding
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        strides: list[int] = [8, 16, 32],
        in_channels: list[int] = [256, 512, 1024],
        act: str = "silu",
        depthwise: bool = False,
        # RTMDet assignment config
        soft_center_radius: float = 3.0,
        iou_cost_weight: float = 3.0,
        # QFL config
        qfl_beta: float = 2.0,
        # DFL config (Phase 2)
        use_dfl: bool = False,
        reg_max: int = 16,
        dfl_weight: float = 0.25,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        # RTMDet config
        self.soft_center_radius = soft_center_radius
        self.iou_cost_weight = iou_cost_weight
        self.qfl_beta = qfl_beta

        # DFL config
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.dfl_weight = dfl_weight
        reg_out_channels = 4 * (reg_max + 1) if use_dfl else 4

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=reg_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.qfl_loss = QualityFocalLoss(beta=qfl_beta, reduction="none")
        self.iou_loss = IOUloss(reduction="none", loss_type="giou")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        # DFL modules
        if use_dfl:
            self.integral = Integral(reg_max)
            self.dfl_loss = DistributionFocalLoss(reduction="none")

    def initialize_biases(self, prior_prob: float) -> None:
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1 or self.use_dfl:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_ch = 4 * (self.reg_max + 1) if self.use_dfl else 4
                    reg_output = reg_output.view(
                        batch_size, 1, reg_ch, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, reg_ch
                    )
                    origin_preds.append(reg_output.clone())

            else:
                if self.use_dfl:
                    # Decode distribution to 4 distance values, then to cxcywh
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    # reg_output: (B, 4*(reg_max+1), H, W) -> (B, H*W, 4*(reg_max+1))
                    reg_flat = reg_output.permute(0, 2, 3, 1).reshape(
                        batch_size, hsize * wsize, -1
                    )
                    # Integral: (B*H*W, 4*(reg_max+1)) -> (B*H*W, 4) ltrb distances
                    ltrb = self.integral(reg_flat.reshape(-1, 4 * (self.reg_max + 1)))
                    ltrb = ltrb.reshape(batch_size, hsize * wsize, 4)
                    # Convert ltrb to cxcywh using grid centers
                    grid = self.grids[k]
                    if grid.shape[2:4] != reg_output.shape[2:4]:
                        yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
                        grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(xin[0].type())
                        self.grids[k] = grid
                    grid_flat = grid.view(1, -1, 2)
                    anchor_centers = (grid_flat + 0.5) * stride_this_level
                    # Expand for batch
                    anchor_centers = anchor_centers.expand(batch_size, -1, -1)
                    x1 = anchor_centers[..., 0:1] - ltrb[..., 0:1] * stride_this_level
                    y1 = anchor_centers[..., 1:2] - ltrb[..., 1:2] * stride_this_level
                    x2 = anchor_centers[..., 0:1] + ltrb[..., 2:3] * stride_this_level
                    y2 = anchor_centers[..., 1:2] + ltrb[..., 3:4] * stride_this_level
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = (x2 - x1).clamp(min=0)
                    h = (y2 - y1).clamp(min=0)
                    # (B, H*W, 4) in cxcywh -> (B, 4, H, W) to match expected layout
                    decoded_reg = torch.cat([cx, cy, w, h], dim=-1)
                    decoded_reg = decoded_reg.permute(0, 2, 1).reshape(
                        batch_size, 4, hsize, wsize
                    )
                    output = torch.cat(
                        [decoded_reg, obj_output.sigmoid(), cls_output.sigmoid()], 1
                    )
                else:
                    output = torch.cat(
                        [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                    )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        reg_ch = 4 * (self.reg_max + 1) if self.use_dfl else 4
        n_ch = reg_ch + 1 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)

        if self.use_dfl:
            # DFL path: decode distribution -> ltrb distances -> cxcywh
            reg_logits = output[..., :reg_ch]  # (B, N, 4*(reg_max+1))
            ltrb = self.integral(reg_logits.reshape(-1, reg_ch))
            ltrb = ltrb.reshape(batch_size, -1, 4)  # (B, N, 4) in stride units
            anchor_centers = (grid + 0.5) * stride  # (1, N, 2) absolute pixels
            decoded_boxes = _distance2bbox(
                anchor_centers.reshape(-1, 2).expand(batch_size * ltrb.shape[1], -1)
                if batch_size == 1
                else anchor_centers.expand(batch_size, -1, -1).reshape(-1, 2),
                ltrb.reshape(-1, 4),
                torch.full((ltrb.reshape(-1, 4).shape[0],), stride, device=ltrb.device, dtype=ltrb.dtype),
            ).reshape(batch_size, -1, 4)
            # Replace the reg channels with decoded cxcywh
            output = torch.cat([decoded_boxes, output[..., reg_ch:]], dim=-1)
        else:
            output[..., :2] = (output[..., :2] + grid) * stride
            output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        return output, grid

    def decode_outputs(self, outputs, dtype):
        """Decode inference outputs. With DFL, boxes are already decoded in get_output_and_grid/forward."""
        if self.use_dfl:
            # Boxes already decoded to cxcywh absolute coords in forward()
            return outputs

        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4] — decoded cxcywh
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1 or self.use_dfl:
            origin_preds = torch.cat(origin_preds, 1)

        cls_labels = []   # per-anchor class index (-1 for background)
        cls_scores = []   # per-anchor IoU quality score (0 for background)
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        iou_weights = []
        dfl_targets = []  # ltrb distance targets for DFL

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_label = outputs.new_full((total_num_anchors,), -1, dtype=torch.long)
                cls_score = outputs.new_zeros((total_num_anchors,))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                iou_weight = outputs.new_zeros((0,))
                dfl_target = outputs.new_zeros((0, 4))
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory. " not in str(e):
                        raise

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                # QFL targets for ALL anchors: (label_index, iou_score)
                # Background anchors get label=-1, score=0 (handled by QFL)
                cls_label = outputs.new_full((total_num_anchors,), -1, dtype=torch.long)
                cls_score = outputs.new_zeros((total_num_anchors,))
                cls_label[fg_mask] = gt_matched_classes.to(torch.long)
                cls_score[fg_mask] = pred_ious_this_matching

                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                iou_weight = pred_ious_this_matching

                # Compute DFL distance targets for positive anchors
                if self.use_dfl:
                    anchor_cx = (x_shifts[0][fg_mask] + 0.5) * expanded_strides[0][fg_mask]
                    anchor_cy = (y_shifts[0][fg_mask] + 0.5) * expanded_strides[0][fg_mask]
                    anchor_points = torch.stack([anchor_cx, anchor_cy], dim=-1)
                    dfl_target = _bbox2distance(
                        anchor_points,
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        self.reg_max,
                    )
                else:
                    dfl_target = outputs.new_zeros((num_fg_img, 4))

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_labels.append(cls_label)
            cls_scores.append(cls_score)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            iou_weights.append(iou_weight)
            dfl_targets.append(dfl_target)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_labels = torch.cat(cls_labels, 0)   # (B*N,) with -1 for background
        cls_scores = torch.cat(cls_scores, 0)   # (B*N,) IoU scores (0 for bg)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        iou_weights = torch.cat(iou_weights, 0)
        dfl_targets = torch.cat(dfl_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        # IoU-weighted regression loss (RTMDet style)
        raw_iou_loss = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        loss_iou = (raw_iou_loss * iou_weights).sum() / num_fg

        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        # QualityFocalLoss on ALL anchors (fg + bg), matching RTMDet/mmyolo.
        # Background anchors get BCE(logit, 0) * sigmoid^beta suppression.
        # Foreground anchors get BCE(logit, IoU) * |IoU - sigmoid|^beta.
        all_cls_preds = cls_preds.view(-1, self.num_classes)
        loss_cls = self._quality_focal_loss(
            all_cls_preds, cls_labels, cls_scores, self.qfl_beta,
        ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        # DFL loss on raw distribution logits
        if self.use_dfl and fg_masks.any():
            reg_logits_fg = origin_preds.view(-1, 4 * (self.reg_max + 1))[fg_masks]
            reg_logits_per_side = reg_logits_fg.reshape(-1, self.reg_max + 1)
            dfl_targets_flat = dfl_targets.reshape(-1)
            raw_dfl_loss = self.dfl_loss(reg_logits_per_side, dfl_targets_flat)
            raw_dfl_loss = raw_dfl_loss.reshape(-1, 4).sum(dim=-1)
            loss_dfl = (raw_dfl_loss * iou_weights).sum() / num_fg
        else:
            loss_dfl = 0.0

        reg_weight = 2.0  # RTMDet uses 2.0 (YOLOX used 5.0 with IoU^2 loss)
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1 + self.dfl_weight * loss_dfl

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    @staticmethod
    def _quality_focal_loss(
        pred: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        beta: float = 2.0,
    ) -> torch.Tensor:
        """QFL on all anchors matching mmyolo/RTMDet formulation.

        1. All anchors get background loss: BCE(logit, 0) * sigmoid^beta
        2. Foreground anchors (label >= 0) get overwritten with:
           BCE(logit[label], IoU) * |IoU - sigmoid(logit[label])|^beta

        Args:
            pred: Class logits ``(N, C)``.
            labels: Per-anchor class index ``(N,)``, -1 for background.
            scores: Per-anchor IoU quality score ``(N,)``, 0 for background.
            beta: Focal modulation exponent.

        Returns:
            Per-anchor loss ``(N,)`` summed over classes.
        """
        # Compute in float32 for numerical stability (AMP sends half-precision)
        pred = pred.float()
        pred_sigmoid = pred.sigmoid()
        # Step 1: background loss for all anchors, all classes
        zerolabel = pred.new_zeros(pred.shape)
        loss = F.binary_cross_entropy_with_logits(
            pred, zerolabel, reduction="none",
        ) * pred_sigmoid.pow(beta)

        # Step 2: overwrite foreground positions with soft QFL
        pos_mask = labels >= 0
        if pos_mask.any():
            pos_inds = pos_mask.nonzero(as_tuple=False).squeeze(1)
            pos_labels = labels[pos_inds]
            pos_scores = scores[pos_inds].float()

            scale = (pos_scores - pred_sigmoid[pos_inds, pos_labels]).abs().pow(beta)
            loss[pos_inds, pos_labels] = F.binary_cross_entropy_with_logits(
                pred[pos_inds, pos_labels], pos_scores, reduction="none",
            ) * scale

        # Sum over classes -> per-anchor scalar
        return loss.sum(dim=-1)

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):
        if mode == "cpu":
            logger.warning("Using CPU for the current batch due to memory constraints")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        # Soft center prior (RTMDet)
        fg_mask, soft_center_prior = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious = pair_wise_ious.clamp(min=0.0, max=1.0)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        # RTMDet soft classification cost on GT class only (mmyolo parity).
        with torch.amp.autocast("cuda", enabled=False):
            cls_logits = cls_preds_.float()
            obj_logits = obj_preds_.float()
            pred_scores = (cls_logits.sigmoid() * obj_logits.sigmoid()).sqrt()

            # Extract GT class channel for each GT: [num_gt, num_anchors]
            gt_class_inds = gt_classes.long()
            pairwise_pred_scores = pred_scores[:, gt_class_inds].T
            pairwise_pred_scores = pairwise_pred_scores.clamp(min=1e-6, max=1.0 - 1e-6)

            soft_target = pair_wise_ious

            scale_factor = (soft_target - pairwise_pred_scores).abs().pow(2.0)
            pair_wise_cls_loss = (
                F.binary_cross_entropy(
                    pairwise_pred_scores,
                    soft_target,
                    reduction="none",
                ) * scale_factor
            )
        del cls_logits, obj_logits, pred_scores, pairwise_pred_scores, scale_factor

        cost = (
            pair_wise_cls_loss
            + self.iou_cost_weight * pair_wise_ious_loss
            + soft_center_prior
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self,
        gt_bboxes_per_image: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RTMDet-style soft center prior with inside-GT-box pre-filtering.

        Pre-filters anchors to those geometrically inside at least one GT box
        (matching mmdetection's ``DynamicSoftLabelAssigner``), then applies a
        continuous exponential cost ``10^(distance/stride - radius)`` that
        penalizes anchors far from GT centers.

        Args:
            gt_bboxes_per_image: GT boxes ``[num_gt, 4]`` in cxcywh format.
            expanded_strides: Per-anchor strides ``[1, n_anchors]``.
            x_shifts: Grid x-coordinates ``[1, n_anchors]``.
            y_shifts: Grid y-coordinates ``[1, n_anchors]``.

        Returns:
            anchor_filter: Boolean mask ``[n_anchors]`` of candidate anchors.
            soft_center_prior: Continuous cost ``[num_gt, num_filtered]``.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers = (x_shifts[0] + 0.5) * expanded_strides_per_image
        y_centers = (y_shifts[0] + 0.5) * expanded_strides_per_image

        # GT boxes: cxcywh -> xyxy edges
        gt_x1 = gt_bboxes_per_image[:, 0:1] - gt_bboxes_per_image[:, 2:3] / 2
        gt_y1 = gt_bboxes_per_image[:, 1:2] - gt_bboxes_per_image[:, 3:4] / 2
        gt_x2 = gt_bboxes_per_image[:, 0:1] + gt_bboxes_per_image[:, 2:3] / 2
        gt_y2 = gt_bboxes_per_image[:, 1:2] + gt_bboxes_per_image[:, 3:4] / 2

        # Inside-GT-box check: anchor center must be inside at least one GT box
        left = x_centers.unsqueeze(0) - gt_x1
        top = y_centers.unsqueeze(0) - gt_y1
        right = gt_x2 - x_centers.unsqueeze(0)
        bottom = gt_y2 - y_centers.unsqueeze(0)
        deltas = torch.stack([left, top, right, bottom], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
        anchor_filter = is_in_gts.any(dim=0)

        # Fallback: if no anchor is inside any GT box (tiny objects), use
        # nearest anchors by distance to ensure at least some candidates
        if not anchor_filter.any():
            gt_cx = gt_bboxes_per_image[:, 0:1]
            gt_cy = gt_bboxes_per_image[:, 1:2]
            all_dist = torch.sqrt(
                (x_centers.unsqueeze(0) - gt_cx) ** 2
                + (y_centers.unsqueeze(0) - gt_cy) ** 2
            )
            min_dist_per_anchor = all_dist.min(dim=0).values
            # Keep the closest anchors (within 3 stride units of nearest GT)
            threshold = expanded_strides_per_image * self.soft_center_radius
            anchor_filter = min_dist_per_anchor < threshold

        # Soft center prior: 10^(distance/stride - radius)
        gt_cx = gt_bboxes_per_image[:, 0:1]
        gt_cy = gt_bboxes_per_image[:, 1:2]
        distance = torch.sqrt(
            (x_centers[anchor_filter].unsqueeze(0) - gt_cx) ** 2
            + (y_centers[anchor_filter].unsqueeze(0) - gt_cy) ** 2
        ) / expanded_strides_per_image[anchor_filter].unsqueeze(0)

        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        return anchor_filter, soft_center_prior

    def simota_matching(
        self,
        cost: torch.Tensor,
        pair_wise_ious: torch.Tensor,
        gt_classes: torch.Tensor,
        num_gt: int,
        fg_mask: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(13, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
