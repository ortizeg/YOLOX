#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
# DINO-X Head: RTMDet-style head for YOLOX — no objectness branch.

from __future__ import annotations

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, meshgrid

from .losses import IOUloss, QualityFocalLoss
from .network_blocks import BaseConv, DWConv


class DINOXHead(nn.Module):
    """RTMDet-style detection head for YOLOX backbone.

    Key difference from YOLOXHead: **no objectness branch**. Classification
    directly predicts joint class+quality scores, trained with QFL. This
    matches RTMDet's architecture where cls score alone determines detection
    confidence.

    Changes from YOLOXHead:
    1. No obj_preds / loss_obj — cls handles objectness via QFL
    2. QFL on ALL anchors (bg suppression + fg soft targets)
    3. Soft center prior with inside-GT-box filtering
    4. GIoU loss with IoU-weighted regression
    5. Assigner cost on GT class only (not all C classes)
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
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        # RTMDet config
        self.soft_center_radius = soft_center_radius
        self.iou_cost_weight = iou_cost_weight
        self.qfl_beta = qfl_beta

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1, stride=1, act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    Conv(in_channels=int(256 * width), out_channels=int(256 * width),
                         ksize=3, stride=1, act=act),
                    Conv(in_channels=int(256 * width), out_channels=int(256 * width),
                         ksize=3, stride=1, act=act),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    Conv(in_channels=int(256 * width), out_channels=int(256 * width),
                         ksize=3, stride=1, act=act),
                    Conv(in_channels=int(256 * width), out_channels=int(256 * width),
                         ksize=3, stride=1, act=act),
                )
            )
            self.cls_preds.append(
                nn.Conv2d(int(256 * width), self.num_classes, 1, 1, 0)
            )
            self.reg_preds.append(
                nn.Conv2d(int(256 * width), 4, 1, 1, 0)
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.iou_loss = IOUloss(reduction="none", loss_type="giou")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob: float) -> None:
        for conv in self.cls_preds:
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

            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)

            if self.training:
                # Training: [reg(4), cls(C)] — no obj channel
                output = torch.cat([reg_output, cls_output], 1)
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
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
            else:
                # Inference: [reg(4), obj=1.0(1), cls_sigmoid(C)]
                # Insert constant obj=1.0 for compatibility with postprocess
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                ones = torch.ones(batch_size, 1, hsize, wsize,
                                  device=reg_output.device, dtype=reg_output.dtype)
                output = torch.cat(
                    [reg_output, ones, cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs, x_shifts, y_shifts, expanded_strides,
                labels, torch.cat(outputs, 1), origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
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
        n_ch = 4 + self.num_classes  # no obj channel in training
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
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
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
            outputs[..., 4:]  # obj=1.0 + cls already applied in forward
        ], dim=-1)
        return outputs

    def get_losses(
        self, imgs, x_shifts, y_shifts, expanded_strides,
        labels, outputs, origin_preds, dtype,
    ):
        bbox_preds = outputs[:, :, :4]       # [batch, n_anchors, 4]
        cls_preds = outputs[:, :, 4:]        # [batch, n_anchors, C] — no obj column

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_labels = []
        cls_scores = []
        reg_targets = []
        l1_targets = []
        fg_masks = []
        iou_weights = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_label = outputs.new_full((total_num_anchors,), -1, dtype=torch.long)
                cls_score = torch.zeros(total_num_anchors, device=outputs.device, dtype=torch.float32)
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                iou_weight = outputs.new_zeros((0,))
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
                        x_shifts, y_shifts, cls_preds,
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
                        x_shifts, y_shifts, cls_preds, "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_label = outputs.new_full((total_num_anchors,), -1, dtype=torch.long)
                cls_score = torch.zeros(total_num_anchors, device=outputs.device, dtype=torch.float32)
                cls_label[fg_mask] = gt_matched_classes.to(torch.long)
                cls_score[fg_mask] = pred_ious_this_matching

                reg_target = gt_bboxes_per_image[matched_gt_inds]
                iou_weight = pred_ious_this_matching
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
            fg_masks.append(fg_mask)
            iou_weights.append(iou_weight)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_labels = torch.cat(cls_labels, 0)
        cls_scores = torch.cat(cls_scores, 0)
        reg_targets = torch.cat(reg_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        iou_weights = torch.cat(iou_weights, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        # GIoU regression loss, IoU-weighted
        raw_iou_loss = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        loss_iou = (raw_iou_loss * iou_weights).sum() / num_fg

        # QFL on ALL anchors — no separate obj loss
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

        reg_weight = 2.0
        loss = reg_weight * loss_iou + loss_cls + loss_l1

        # Return 6 values for compatibility (loss_obj = 0)
        return (
            loss,
            reg_weight * loss_iou,
            torch.tensor(0.0, device=loss.device),  # no obj loss
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
        """QFL on all anchors matching mmyolo/RTMDet formulation."""
        pred = pred.float()
        pred_sigmoid = pred.sigmoid()
        zerolabel = pred.new_zeros(pred.shape)
        loss = F.binary_cross_entropy_with_logits(
            pred, zerolabel, reduction="none",
        ) * pred_sigmoid.pow(beta)

        pos_mask = labels >= 0
        if pos_mask.any():
            pos_inds = pos_mask.nonzero(as_tuple=False).squeeze(1)
            pos_labels = labels[pos_inds].clamp(0, pred.shape[1] - 1)
            pos_scores = scores[pos_inds].float().clamp(0.0, 1.0)

            scale = (pos_scores - pred_sigmoid[pos_inds, pos_labels]).abs().pow(beta)
            loss[pos_inds, pos_labels] = F.binary_cross_entropy_with_logits(
                pred[pos_inds, pos_labels], pos_scores, reduction="none",
            ) * scale

        return loss.sum(dim=-1)

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self, batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
        bboxes_preds_per_image, expanded_strides, x_shifts, y_shifts,
        cls_preds, mode="gpu",
    ):
        if mode == "cpu":
            logger.warning("Using CPU for the current batch due to memory constraints")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, soft_center_prior = self.get_geometry_constraint(
            gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious = torch.nan_to_num(pair_wise_ious, nan=0.0).clamp(min=0.0, max=1.0)

        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_ = cls_preds_.cpu()

        # RTMDet cost: GT class only, raw cls logits (no obj branch)
        with torch.amp.autocast("cuda", enabled=False):
            cls_logits = cls_preds_.float()
            pred_scores = cls_logits.sigmoid()

            gt_class_inds = gt_classes.long().clamp(0, self.num_classes - 1)
            pairwise_pred_scores = pred_scores[:, gt_class_inds].T
            pairwise_pred_scores = pairwise_pred_scores.clamp(min=1e-6, max=1.0 - 1e-6)

            soft_target = pair_wise_ious

            scale_factor = (soft_target - pairwise_pred_scores).abs().pow(2.0)
            pair_wise_cls_loss = (
                F.binary_cross_entropy(
                    pairwise_pred_scores, soft_target, reduction="none",
                ) * scale_factor
            )
        del cls_logits, pred_scores, pairwise_pred_scores, scale_factor

        cost = (
            pair_wise_cls_loss
            + self.iou_cost_weight * pair_wise_ious_loss
            + soft_center_prior
        )

        (
            num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg)

    def get_geometry_constraint(
        self, gt_bboxes_per_image: torch.Tensor,
        expanded_strides: torch.Tensor,
        x_shifts: torch.Tensor, y_shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RTMDet-style soft center prior with inside-GT-box pre-filtering."""
        expanded_strides_per_image = expanded_strides[0]
        x_centers = (x_shifts[0] + 0.5) * expanded_strides_per_image
        y_centers = (y_shifts[0] + 0.5) * expanded_strides_per_image

        gt_x1 = gt_bboxes_per_image[:, 0:1] - gt_bboxes_per_image[:, 2:3] / 2
        gt_y1 = gt_bboxes_per_image[:, 1:2] - gt_bboxes_per_image[:, 3:4] / 2
        gt_x2 = gt_bboxes_per_image[:, 0:1] + gt_bboxes_per_image[:, 2:3] / 2
        gt_y2 = gt_bboxes_per_image[:, 1:2] + gt_bboxes_per_image[:, 3:4] / 2

        left = x_centers.unsqueeze(0) - gt_x1
        top = y_centers.unsqueeze(0) - gt_y1
        right = gt_x2 - x_centers.unsqueeze(0)
        bottom = gt_y2 - y_centers.unsqueeze(0)
        deltas = torch.stack([left, top, right, bottom], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
        anchor_filter = is_in_gts.any(dim=0)

        if not anchor_filter.any():
            gt_cx = gt_bboxes_per_image[:, 0:1]
            gt_cy = gt_bboxes_per_image[:, 1:2]
            all_dist = torch.sqrt(
                (x_centers.unsqueeze(0) - gt_cx) ** 2
                + (y_centers.unsqueeze(0) - gt_cy) ** 2
            )
            min_dist_per_anchor = all_dist.min(dim=0).values
            threshold = expanded_strides_per_image * self.soft_center_radius
            anchor_filter = min_dist_per_anchor < threshold

        gt_cx = gt_bboxes_per_image[:, 0:1]
        gt_cy = gt_bboxes_per_image[:, 1:2]
        distance = torch.sqrt(
            (x_centers[anchor_filter].unsqueeze(0) - gt_cx) ** 2
            + (y_centers[anchor_filter].unsqueeze(0) - gt_cy) ** 2
        ) / expanded_strides_per_image[anchor_filter].unsqueeze(0)

        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        return anchor_filter, soft_center_prior

    def simota_matching(
        self, cost: torch.Tensor, pair_wise_ious: torch.Tensor,
        gt_classes: torch.Tensor, num_gt: int, fg_mask: torch.Tensor,
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
