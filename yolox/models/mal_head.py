#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# Matchability-Aware Loss (MAL) head — adapts DEIM's MAL for YOLOX.

from __future__ import annotations

import torch
import torch.nn.functional as F

from loguru import logger

from .yolo_head import YOLOXHead


class YOLOXHeadMAL(YOLOXHead):
    """YOLOXHead with Matchability-Aware Loss (MAL) for classification.

    Adapts DEIM's MAL (arXiv 2412.04234) for YOLOX. Instead of using IoU
    as the soft classification target, MAL uses a matchability score that
    combines localization AND classification quality:

        matchability = IoU^gamma * cls_score^(1 - gamma)

    This replaces the classification TARGET (not the loss function):
    - Standard YOLOX: cls_target = one_hot * IoU
    - MAL:            cls_target = one_hot * matchability

    The target is self-referential — it adapts during training based on
    the model's current prediction quality. This teaches the model to
    output calibrated confidence scores that reflect match quality.

    Args:
        mal_gamma: Controls IoU vs cls_score balance. Default ``1.5``
            (DEIM default). Higher gamma → more weight on IoU quality.
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        strides: list[int] = [8, 16, 32],
        in_channels: list[int] = [256, 512, 1024],
        act: str = "silu",
        depthwise: bool = False,
        mal_gamma: float = 1.5,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            width=width,
            strides=strides,
            in_channels=in_channels,
            act=act,
            depthwise=depthwise,
        )
        self.mal_gamma = mal_gamma

    def get_losses(
        self, imgs, x_shifts, y_shifts, expanded_strides,
        labels, outputs, origin_preds, dtype,
    ):
        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4:5]
        cls_preds = outputs[:, :, 5:]

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes, fg_mask,
                        pred_ious_this_matching, matched_gt_inds, num_fg_img,
                    ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides,
                        x_shifts, y_shifts, cls_preds, obj_preds,
                    )
                except RuntimeError as e:
                    if "CUDA out of memory. " not in str(e):
                        raise
                    logger.error("OOM in label assignment, falling back to CPU")
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes, fg_mask,
                        pred_ious_this_matching, matched_gt_inds, num_fg_img,
                    ) = self.get_assignments(
                        batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                        bboxes_preds_per_image, expanded_strides,
                        x_shifts, y_shifts, cls_preds, obj_preds, "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                # MAL: compute matchability as the soft classification target
                # matchability = IoU^gamma * cls_score^(1 - gamma)
                with torch.no_grad():
                    fg_cls_preds = cls_preds[batch_idx][fg_mask]
                    fg_obj_preds = obj_preds[batch_idx][fg_mask]

                    cls_scores_fg = (
                        fg_cls_preds.float().sigmoid() * fg_obj_preds.float().sigmoid()
                    ).sqrt()

                    gt_cls_inds = gt_matched_classes.long().clamp(0, self.num_classes - 1)
                    matched_cls_score = cls_scores_fg[
                        torch.arange(num_fg_img, device=cls_scores_fg.device), gt_cls_inds
                    ].clamp(min=1e-6)

                    iou_quality = pred_ious_this_matching.clamp(min=1e-6)

                    matchability = (
                        iou_quality.pow(self.mal_gamma)
                        * matched_cls_score.pow(1.0 - self.mal_gamma)
                    ).clamp(min=0.0, max=1.0)

                # Use matchability as target instead of raw IoU
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * matchability.unsqueeze(-1)

                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg

        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        # Standard BCE with matchability-modified targets
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )
