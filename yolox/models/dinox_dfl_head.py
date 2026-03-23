#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# DINOXHead + DFL: Distribution Focal Loss for box regression.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from yolox.utils import bboxes_iou, meshgrid

from .dinox_head import DINOXHead
from .losses import DistributionFocalLoss, Integral
from .network_blocks import BaseConv, DWConv


def _distance2bbox(points: torch.Tensor, distance: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
    """Decode (l, t, r, b) distances to cxcywh boxes."""
    stride = stride.reshape(-1, 1) if stride.dim() == 1 else stride
    x1 = points[:, 0:1] - distance[:, 0:1] * stride
    y1 = points[:, 1:2] - distance[:, 1:2] * stride
    x2 = points[:, 0:1] + distance[:, 2:3] * stride
    y2 = points[:, 1:2] + distance[:, 3:4] * stride
    return torch.cat([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)


def _bbox2distance(points: torch.Tensor, bbox: torch.Tensor, stride: torch.Tensor, reg_max: float) -> torch.Tensor:
    """Encode cxcywh boxes to (l, t, r, b) distances."""
    stride = stride.reshape(-1, 1) if stride.dim() == 1 else stride
    x1 = bbox[:, 0:1] - bbox[:, 2:3] / 2
    y1 = bbox[:, 1:2] - bbox[:, 3:4] / 2
    x2 = bbox[:, 0:1] + bbox[:, 2:3] / 2
    y2 = bbox[:, 1:2] + bbox[:, 3:4] / 2
    left = (points[:, 0:1] - x1) / stride
    top = (points[:, 1:2] - y1) / stride
    right = (x2 - points[:, 0:1]) / stride
    bottom = (y2 - points[:, 1:2]) / stride
    return torch.cat([left, top, right, bottom], dim=-1).clamp(min=0, max=reg_max - 0.01)


class DINOXHeadDFL(DINOXHead):
    """DINOXHead with Distribution Focal Loss for box regression.

    Replaces 4-scalar regression with 4*(reg_max+1) distribution output.
    Inherits all DINOXHead features (no obj, QFL, soft center prior, GIoU).
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        strides: list[int] = [8, 16, 32],
        in_channels: list[int] = [256, 512, 1024],
        act: str = "silu",
        depthwise: bool = False,
        soft_center_radius: float = 3.0,
        iou_cost_weight: float = 3.0,
        qfl_beta: float = 2.0,
        reg_max: int = 16,
        dfl_weight: float = 0.25,
    ) -> None:
        # Init parent without reg_preds (we'll override them)
        super().__init__(
            num_classes=num_classes, width=width, strides=strides,
            in_channels=in_channels, act=act, depthwise=depthwise,
            soft_center_radius=soft_center_radius,
            iou_cost_weight=iou_cost_weight, qfl_beta=qfl_beta,
        )
        self.reg_max = reg_max
        self.dfl_weight = dfl_weight

        # Replace reg_preds: 4 -> 4*(reg_max+1)
        reg_out = 4 * (reg_max + 1)
        self.reg_preds = nn.ModuleList()
        for i in range(len(in_channels)):
            self.reg_preds.append(
                nn.Conv2d(int(256 * width), reg_out, 1, 1, 0)
            )

        self.integral = Integral(reg_max)
        self.dfl_loss = DistributionFocalLoss(reduction="none")

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        raw_reg_preds = []  # raw distribution logits for DFL loss
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)  # (B, 4*(reg_max+1), H, W)

            if self.training:
                # Save raw reg logits for DFL loss
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_ch = 4 * (self.reg_max + 1)
                raw = reg_output.view(batch_size, 1, reg_ch, hsize, wsize)
                raw = raw.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, reg_ch)
                raw_reg_preds.append(raw.clone())

                # Decode distribution -> ltrb -> cxcywh for IoU loss + assignment
                ltrb = self.integral(reg_output.permute(0, 2, 3, 1).reshape(-1, reg_ch))
                ltrb = ltrb.reshape(batch_size, hsize * wsize, 4)

                grid = self.grids[k]
                if grid.shape[2:4] != reg_output.shape[2:4]:
                    yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
                    grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(xin[0].type())
                    self.grids[k] = grid
                grid_flat = grid.view(1, -1, 2)
                anchor_centers = (grid_flat + 0.5) * stride_this_level

                decoded = _distance2bbox(
                    anchor_centers.expand(batch_size, -1, -1).reshape(-1, 2),
                    ltrb.reshape(-1, 4),
                    torch.full((batch_size * hsize * wsize,), stride_this_level,
                               device=ltrb.device, dtype=ltrb.dtype),
                ).reshape(batch_size, hsize * wsize, 4)

                output = torch.cat([decoded, cls_output.permute(0, 2, 3, 1).reshape(
                    batch_size, hsize * wsize, -1)], dim=-1)

                x_shifts.append(grid_flat[:, :, 0])
                y_shifts.append(grid_flat[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, hsize * wsize).fill_(stride_this_level).type_as(xin[0])
                )
            else:
                # Inference: decode distribution -> cxcywh, insert obj=1.0
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_ch = 4 * (self.reg_max + 1)
                ltrb = self.integral(reg_output.permute(0, 2, 3, 1).reshape(-1, reg_ch))
                ltrb = ltrb.reshape(batch_size, hsize * wsize, 4)

                grid = self.grids[k]
                if grid.shape[2:4] != reg_output.shape[2:4]:
                    yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
                    grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(xin[0].type())
                    self.grids[k] = grid
                grid_flat = grid.view(1, -1, 2)
                anchor_centers = (grid_flat + 0.5) * stride_this_level

                decoded = _distance2bbox(
                    anchor_centers.expand(batch_size, -1, -1).reshape(-1, 2),
                    ltrb.reshape(-1, 4),
                    torch.full((batch_size * hsize * wsize,), stride_this_level,
                               device=ltrb.device, dtype=ltrb.dtype),
                ).reshape(batch_size, hsize * wsize, 4)

                ones = torch.ones(batch_size, hsize * wsize, 1,
                                  device=decoded.device, dtype=decoded.dtype)
                cls_sig = cls_output.permute(0, 2, 3, 1).reshape(
                    batch_size, hsize * wsize, -1).sigmoid()
                output = torch.cat([decoded, ones, cls_sig], dim=-1)
                # Reshape to (B, 5+C, H*W) for consistency with cat later
                output = output.permute(0, 2, 1).reshape(batch_size, -1, hsize, wsize)

            outputs.append(output)

        if self.training:
            all_outputs = torch.cat(outputs, 1)  # (B, N_total, 4+C)
            all_raw_regs = torch.cat(raw_reg_preds, 1)  # (B, N_total, 68)
            return self.get_losses(
                imgs, x_shifts, y_shifts, expanded_strides,
                labels, all_outputs, all_raw_regs, dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            # Already decoded — just return
            return outputs

    def get_losses(self, imgs, x_shifts, y_shifts, expanded_strides,
                   labels, outputs, raw_reg_preds, dtype):
        bbox_preds = outputs[:, :, :4]
        cls_preds = outputs[:, :, 4:]

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_labels = []
        cls_scores = []
        reg_targets = []
        fg_masks = []
        iou_weights = []
        dfl_targets = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_label = outputs.new_full((total_num_anchors,), -1, dtype=torch.long)
                cls_score = torch.zeros(total_num_anchors, device=outputs.device, dtype=torch.float32)
                reg_target = outputs.new_zeros((0, 4))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                iou_weight = outputs.new_zeros((0,))
                dfl_target = outputs.new_zeros((0, 4))
            else:
                gt_bboxes = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bbox_preds_img = bbox_preds[batch_idx]

                try:
                    (gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg_img,
                     ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes, gt_classes,
                        bbox_preds_img, expanded_strides, x_shifts, y_shifts, cls_preds,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory. " not in str(e):
                        raise
                    logger.error("OOM, falling back to CPU")
                    torch.cuda.empty_cache()
                    (gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg_img,
                     ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes, gt_classes,
                        bbox_preds_img, expanded_strides, x_shifts, y_shifts, cls_preds, "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_label = outputs.new_full((total_num_anchors,), -1, dtype=torch.long)
                cls_score = torch.zeros(total_num_anchors, device=outputs.device, dtype=torch.float32)
                cls_label[fg_mask] = gt_matched_classes.to(torch.long)
                cls_score[fg_mask] = pred_ious

                reg_target = gt_bboxes[matched_gt_inds]
                iou_weight = pred_ious

                # DFL distance targets
                anchor_cx = (x_shifts[0][fg_mask] + 0.5) * expanded_strides[0][fg_mask]
                anchor_cy = (y_shifts[0][fg_mask] + 0.5) * expanded_strides[0][fg_mask]
                anchor_points = torch.stack([anchor_cx, anchor_cy], dim=-1)
                dfl_target = _bbox2distance(
                    anchor_points, gt_bboxes[matched_gt_inds],
                    expanded_strides[0][fg_mask], self.reg_max,
                )

            cls_labels.append(cls_label)
            cls_scores.append(cls_score)
            reg_targets.append(reg_target)
            fg_masks.append(fg_mask)
            iou_weights.append(iou_weight)
            dfl_targets.append(dfl_target)

        cls_labels = torch.cat(cls_labels, 0)
        cls_scores = torch.cat(cls_scores, 0)
        reg_targets = torch.cat(reg_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        iou_weights = torch.cat(iou_weights, 0)
        dfl_targets = torch.cat(dfl_targets, 0)

        num_fg = max(num_fg, 1)

        # GIoU loss
        raw_iou_loss = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        loss_iou = (raw_iou_loss * iou_weights).sum() / num_fg

        # QFL on all anchors
        loss_cls = self._quality_focal_loss(
            cls_preds.view(-1, self.num_classes), cls_labels, cls_scores, self.qfl_beta,
        ).sum() / num_fg

        # DFL loss
        if fg_masks.any():
            reg_logits_fg = raw_reg_preds.view(-1, 4 * (self.reg_max + 1))[fg_masks]
            reg_per_side = reg_logits_fg.reshape(-1, self.reg_max + 1)
            dfl_flat = dfl_targets.reshape(-1)
            raw_dfl = self.dfl_loss(reg_per_side, dfl_flat)
            raw_dfl = raw_dfl.reshape(-1, 4).sum(dim=-1)
            loss_dfl = (raw_dfl * iou_weights).sum() / num_fg
        else:
            loss_dfl = 0.0

        reg_weight = 2.0
        loss = reg_weight * loss_iou + loss_cls + self.dfl_weight * loss_dfl

        return (
            loss,
            reg_weight * loss_iou,
            torch.tensor(0.0, device=loss.device),
            loss_cls,
            torch.tensor(0.0, device=loss.device) if not isinstance(loss_dfl, float) else loss_dfl,
            num_fg / max(num_gts, 1),
        )
