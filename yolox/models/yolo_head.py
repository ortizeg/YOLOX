#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou, cxcywh2xyxy, meshgrid, perf, visualize_assign
from yolox.utils.step_audit import stage

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

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
                    out_channels=4,
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
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
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
                if perf.on("head_nosync"):
                    output, grid, stride_row = self.get_output_and_grid_nosync(
                        output, k, stride_this_level, xin[0].dtype, xin[0].device
                    )
                    x_shifts.append(grid[:, :, 0])
                    y_shifts.append(grid[:, :, 1])
                    expanded_strides.append(stride_row)
                else:
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
                if perf.on("head_nosync"):
                    return self.decode_outputs_nosync(outputs, xin[0].dtype, xin[0].device)
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
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

    def _device_grid(self, key, hsize, wsize, stride, dtype, device):
        """Grid [1, H*W, 2] (x, y) and stride row [1, H*W] built directly on ``device`` (perf flag ``head_nosync``),
        cached per (key, H, W, stride, dtype, device) so multiscale sizes are built once each. Same values as the
        CPU-built originals: small integers, exactly representable in every float dtype."""
        cache = self.__dict__.setdefault("_nosync_cache", {})
        ck = (key, hsize, wsize, stride, dtype, device)
        hit = cache.get(ck)
        if hit is None:
            if len(cache) >= 256:  # bound it (11 multiscale sizes x 3 levels in practice)
                cache.clear()
            yv, xv = meshgrid([torch.arange(hsize, device=device), torch.arange(wsize, device=device)])
            grid = torch.stack((xv, yv), 2).view(1, hsize * wsize, 2).to(dtype)
            row = torch.full((1, hsize * wsize), stride, dtype=dtype, device=device)
            hit = cache[ck] = (grid, row)
        return hit

    def get_output_and_grid_nosync(self, output, k, stride, dtype, device):
        """get_output_and_grid without host-to-device copies: grid and stride row come from the device cache."""
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        grid, row = self._device_grid(("train", k), hsize, wsize, stride, dtype, device)
        self.grids[k] = grid.view(1, 1, hsize, wsize, 2)
        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid, row

    def decode_outputs_nosync(self, outputs, dtype, device):
        """decode_outputs with grids and strides built (and cached) on the device."""
        grids = []
        strides = []
        for k, ((hsize, wsize), stride) in enumerate(zip(self.hw, self.strides)):
            grid, row = self._device_grid(("decode", k), int(hsize), int(wsize), stride, dtype, device)
            grids.append(grid)
            strides.append(row.unsqueeze(-1))
        grids = torch.cat(grids, dim=1)
        strides = torch.cat(strides, dim=1)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs

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
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        with stage("head_targets"):  # SimOTA assignment
            counts = self.take_host_label_counts(labels.shape[0]) if perf.on("host_label_counts") else None
            if perf.on("simota_batch"):
                (cls_targets, reg_targets, obj_targets, fg_masks, l1_targets, num_fg,
                 num_gts) = self.get_targets_batched(
                    labels, nlabel, bbox_preds, obj_preds, cls_preds, expanded_strides, x_shifts, y_shifts, dtype,
                    counts=counts)
            else:
                (cls_targets, reg_targets, obj_targets, fg_masks, l1_targets, num_fg,
                 num_gts) = self.get_targets_loop(
                    labels, nlabel, outputs, bbox_preds, obj_preds, cls_preds, expanded_strides, x_shifts, y_shifts,
                    total_num_anchors, dtype, counts=counts)

        if torch.is_tensor(num_fg):  # dense_loss: masked dense losses over every anchor, no host sync
            with stage("head_loss"):
                return self.get_losses_dense(bbox_preds, obj_preds, cls_preds, origin_preds, cls_targets,
                                             reg_targets, obj_targets, fg_masks, l1_targets, num_fg, num_gts)

        with stage("head_loss"):
            num_fg = max(num_fg, 1)
            # simota_nosync returns positive indices (int64) instead of a boolean mask; x[idx] == x[mask] row for row
            loss_iou = (
                self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
            ).sum() / num_fg
            loss_obj = (
                self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
            ).sum() / num_fg
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

    def get_losses_dense(self, bbox_preds, obj_preds, cls_preds, origin_preds, cls_targets, reg_targets,
                         obj_targets, fg, l1_targets, num_fg, num_gts):
        """``dense_loss``: the same four losses as ``get_losses``, summed over every anchor with the positive mask
        instead of over the gathered positives. Positive rows see exactly the inputs they see there; other rows
        contribute exactly 0 to the loss and to every gradient (their IoU inputs are replaced by a unit box, not
        multiplied by 0, so a non-finite prediction on a background anchor cannot turn the sum into NaN). Only the
        summation order differs (tier B)."""
        fgf = fg.to(cls_targets.dtype)
        fg4 = fg[:, None]
        nfg = num_fg.clamp(min=1).to(torch.float32)
        unit = reg_targets.new_ones(())
        pred = torch.where(fg4, bbox_preds.view(-1, 4), unit.to(bbox_preds.dtype))
        tgt = torch.where(fg4, reg_targets, unit)
        loss_iou = (self.iou_loss(pred, tgt) * fgf).sum() / nfg
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / nfg
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes), cls_targets) * fgf[:, None]).sum() / nfg
        if self.use_l1:
            opred = origin_preds.view(-1, 4)
            loss_l1 = (self.l1_loss(opred, torch.where(fg4, l1_targets, opred.detach())) * fgf[:, None]).sum() / nfg
        else:
            loss_l1 = 0.0
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
        return (loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, nfg / max(num_gts, 1))

    def take_host_label_counts(self, batch_size):
        """``host_label_counts``: the per-image label counts the trainer computed from the CPU targets
        (``perf.set_label_counts``), consumed once so a stale list can never be applied to a later batch. Returns
        None (the caller counts on the device, one host sync, as without the flag) when none were provided, e.g.
        a caller that drives the model directly; a wrong-length list is a caller bug and raises."""
        counts = getattr(self, "host_label_counts", None)
        self.host_label_counts = None
        if counts is None:
            if not getattr(self, "_warned_no_host_counts", False):
                self._warned_no_host_counts = True
                logger.warning("perf host_label_counts: no host label counts were set for this step; counting on "
                               "the device instead (perf.set_label_counts before the forward avoids the sync)")
            return None
        if len(counts) != batch_size:
            raise ValueError(f"perf host_label_counts: {len(counts)} host label counts for a batch of {batch_size}")
        return counts

    def get_targets_loop(self, labels, nlabel, outputs, bbox_preds, obj_preds, cls_preds, expanded_strides,
                         x_shifts, y_shifts, total_num_anchors, dtype, counts=None):
        """The original YOLOX per-image target assignment (unchanged apart from being a method). ``counts``: host
        label counts (``host_label_counts``) used instead of reading ``nlabel`` back per image."""
        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx]) if counts is None else counts[batch_idx]
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
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

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

                if not perf.on("no_empty_cache"):
                    torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
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
        return cls_targets, reg_targets, obj_targets, fg_masks, l1_targets, num_fg, num_gts

    @torch.no_grad()
    def get_targets_batched(self, labels, nlabel, bbox_preds, obj_preds, cls_preds, expanded_strides,
                            x_shifts, y_shifts, dtype, counts=None):
        """SimOTA for the whole batch at once (perf flag ``simota_batch``).

        Same element-wise maths as ``get_targets_loop`` / ``get_assignments`` / ``simota_matching``, with ground
        truths padded to the batch maximum and every anchor kept (non-candidates masked) instead of a Python loop
        per image and per ground truth. Two host syncs per batch instead of several per image. Targets come out in
        the same order (image-major, anchor order), so the losses are unchanged.
        """
        B, A = bbox_preds.shape[:2]
        C = self.num_classes
        dev = bbox_preds.device
        if counts is None:
            counts = nlabel.tolist()  # host sync 1 (host_label_counts passes the host list instead)
        num_gts = float(sum(counts))
        G = max(counts) if counts else 0
        if G == 0:
            z = bbox_preds.new_zeros
            fg = torch.zeros(B * A, dtype=torch.bool, device=dev)
            if perf.on("dense_loss"):  # every anchor present, none positive
                return (z((B * A, C)), z((B * A, 4)), z((B * A, 1)).to(dtype), fg, z((B * A, 4)), fg.sum(), num_gts)
            return (z((0, C)), z((0, 4)), z((B * A, 1)).to(dtype), fg, z((0, 4)), 0.0, num_gts)

        gt = labels[:, :G]  # [B, G, 5] = cls, cx, cy, w, h (valid rows first)
        valid = torch.arange(G, device=dev)[None, :] < nlabel[:, None]  # [B, G]
        gt_boxes = gt[..., 1:5]
        gt_classes = gt[..., 0]

        # geometry constraint (get_geometry_constraint), per anchor and gt
        strides = expanded_strides[0]  # [A]
        xc = ((x_shifts[0] + 0.5) * strides).unsqueeze(0)  # [1, A]
        yc = ((y_shifts[0] + 0.5) * strides).unsqueeze(0)
        center_dist = strides.unsqueeze(0) * 1.5
        gl = gt_boxes[..., 0:1] - center_dist  # [B, G, A]
        gr = gt_boxes[..., 0:1] + center_dist
        gtp = gt_boxes[..., 1:2] - center_dist
        gb = gt_boxes[..., 1:2] + center_dist
        center_deltas = torch.stack([xc - gl, yc - gtp, gr - xc, gb - yc], 3)
        is_in_centers = (center_deltas.min(dim=-1).values > 0.0) & valid[..., None]  # [B, G, A]
        del center_deltas, gl, gr, gtp, gb
        cand = is_in_centers.any(dim=1)  # [B, A]  == fg_mask before matching

        # pairwise IoU (bboxes_iou, xyxy=False), same ops broadcast over [B, G, A]
        a = gt_boxes[:, :, None, :]
        b = bbox_preds[:, None, :, :]
        tl = torch.max(a[..., :2] - a[..., 2:] / 2, b[..., :2] - b[..., 2:] / 2)
        br = torch.min(a[..., :2] + a[..., 2:] / 2, b[..., :2] + b[..., 2:] / 2)
        area_a = torch.prod(a[..., 2:], -1)
        area_b = torch.prod(b[..., 2:], -1)
        en = (tl < br).to(tl.dtype).prod(dim=-1)
        area_i = torch.prod(br - tl, -1) * en
        ious = area_i / (area_a + area_b - area_i)  # [B, G, A]
        del tl, br, en, area_i, area_a, area_b
        keep = cand[:, None, :] & valid[..., None]
        ious = torch.where(keep, ious, ious.new_zeros(()))
        ious_loss = -torch.log(ious + 1e-8)

        # class cost
        with torch.amp.autocast("cuda", enabled=False):
            p = (cls_preds.float().sigmoid() * obj_preds.float().sigmoid()).sqrt()  # [B, A, C]
            gcls = gt_classes.to(torch.int64).clamp(0, C - 1)
            if perf.on("cls_cost_algebra"):
                # BCE summed over classes with a one-hot target = sum_k -log(1-p_k) + log(1-p_c) - log(p_c),
                # with the same -100 clamp as F.binary_cross_entropy
                l1m = torch.clamp(torch.log1p(-p), min=-100)  # log(1-p)
                l0 = torch.clamp(torch.log(p), min=-100)
                base = -l1m.sum(-1)  # [B, A]
                idx = gcls[:, :, None].expand(B, G, A)
                pick1 = torch.gather(l1m[:, None].expand(B, G, A, C), 3, idx[..., None])[..., 0]
                pick0 = torch.gather(l0[:, None].expand(B, G, A, C), 3, idx[..., None])[..., 0]
                cls_cost = base[:, None, :] + pick1 - pick0
            else:
                onehot = F.one_hot(gcls, C).float()  # [B, G, C]
                cls_cost = torch.empty((B, G, A), dtype=torch.float32, device=dev)
                for bi in range(B):  # per image keeps the [G, A, C] tensor small; no host sync
                    cls_cost[bi] = F.binary_cross_entropy(
                        p[bi].unsqueeze(0).repeat(G, 1, 1),
                        onehot[bi].unsqueeze(1).repeat(1, A, 1),
                        reduction="none",
                    ).sum(-1)
            del p

        cost = cls_cost + 3.0 * ious_loss + float(1e6) * (~is_in_centers)
        if perf.on("head_nosync"):
            inf = cost.new_full((), float("inf"))  # fill kernel on the device, no host-to-device copy
        else:
            inf = torch.tensor(float("inf"), device=dev)
        cost = torch.where(keep, cost, inf)  # non-candidates and padded gts never match
        del cls_cost, ious_loss

        # dynamic k (simota_matching)
        n_candidate_k = min(10, A)
        topk_ious, _ = torch.topk(ious, n_candidate_k, dim=2)
        dynamic_ks = torch.clamp(topk_ious.sum(2).int(), min=1) * valid  # [B, G]
        if perf.on("simota_nosync"):
            kmax = n_candidate_k  # dynamic_k <= n_candidate_k (sum of <= 10 IoUs <= 1); ranks >= dynamic_k are masked
        else:
            kmax = int(dynamic_ks.max())  # host sync 2
        _, pos = torch.topk(cost, kmax, dim=2, largest=False)  # [B, G, kmax]
        rank_ok = torch.arange(kmax, device=dev)[None, None, :] < dynamic_ks[..., None]
        matching = torch.zeros((B, G, A), dtype=torch.uint8, device=dev)
        matching.scatter_(2, pos, rank_ok.to(torch.uint8))

        # one anchor matched to several gts -> keep the cheapest gt
        anchor_matching_gt = matching.sum(1)  # [B, A]
        multiple = anchor_matching_gt > 1
        cost_argmin = cost.argmin(dim=1)  # [B, A]
        best = F.one_hot(cost_argmin, G).permute(0, 2, 1).to(torch.uint8)  # [B, G, A]
        matching = torch.where(multiple[:, None, :], best, matching)
        fg = anchor_matching_gt > 0  # [B, A]

        fg_flat = fg.reshape(-1)
        matched_gt_inds = matching.argmax(dim=1)  # [B, A]
        pred_ious = (matching * ious).sum(1)  # [B, A]
        if perf.on("dense_loss"):
            # targets for every anchor (image-major, the order of fg_flat), gathered with the matched gt index; rows
            # of non-positive anchors hold arbitrary finite values and are masked out in get_losses. No nonzero(),
            # so no host sync and no data-dependent shape; num_fg stays a device tensor.
            bidx = torch.arange(B, device=dev)[:, None].expand(B, A)
            gt_matched_classes = gt_classes[bidx, matched_gt_inds]  # [B, A]
            cls_targets = F.one_hot(gt_matched_classes.to(torch.int64).clamp(0, C - 1), C) * \
                (pred_ious * fg).unsqueeze(-1)
            cls_targets = cls_targets.reshape(B * A, C)
            reg_targets = gt_boxes[bidx, matched_gt_inds].reshape(B * A, 4)
            obj_targets = fg_flat.unsqueeze(-1).to(dtype)
            l1_targets = None
            if self.use_l1:
                # padded gts (zero boxes) only ever land on non-positive rows; clamp keeps their log finite
                safe = torch.where(fg_flat[:, None], reg_targets, reg_targets.new_ones(()))
                l1_targets = self.get_l1_target(
                    bbox_preds.new_zeros((B * A, 4)),
                    safe,
                    strides.repeat(B),
                    x_shifts=x_shifts[0].repeat(B),
                    y_shifts=y_shifts[0].repeat(B),
                )
            return cls_targets, reg_targets, obj_targets, fg_flat, l1_targets, fg_flat.sum(), num_gts
        if perf.on("simota_nosync"):
            # one nonzero (the only sync after the counts) gives every positive's flat index, in the same
            # image-major anchor order as boolean-mask indexing; everything below gathers with it
            fg_idx = fg_flat.nonzero().squeeze(1)
            mb = torch.div(fg_idx, A, rounding_mode="floor")
            aidx = fg_idx - mb * A
            mg = matched_gt_inds.reshape(-1).index_select(0, fg_idx)
            gt_matched_classes = gt_classes[mb, mg]
            cls_targets = F.one_hot(gt_matched_classes.to(torch.int64), C) * \
                pred_ious.reshape(-1).index_select(0, fg_idx).unsqueeze(-1)
            reg_targets = gt_boxes[mb, mg]
            obj_targets = fg_flat.unsqueeze(-1).to(dtype)
            l1_targets = None
            if self.use_l1:
                l1_targets = self.get_l1_target(
                    bbox_preds.new_zeros((int(fg_idx.numel()), 4)),
                    reg_targets,
                    strides[aidx],
                    x_shifts=x_shifts[0][aidx],
                    y_shifts=y_shifts[0][aidx],
                )
            num_fg = float(fg_idx.numel())  # known on the host after nonzero: no extra sync
            return cls_targets, reg_targets, obj_targets, fg_idx, l1_targets, num_fg, num_gts
        bidx = torch.arange(B, device=dev)[:, None].expand(B, A)
        mg = matched_gt_inds[fg]
        mb = bidx[fg]
        gt_matched_classes = gt_classes[mb, mg]
        cls_targets = F.one_hot(gt_matched_classes.to(torch.int64), C) * pred_ious[fg].unsqueeze(-1)
        reg_targets = gt_boxes[mb, mg]
        obj_targets = fg_flat.unsqueeze(-1).to(dtype)
        l1_targets = None
        if self.use_l1:
            aidx = torch.arange(A, device=dev)[None, :].expand(B, A)[fg]
            l1_targets = self.get_l1_target(
                bbox_preds.new_zeros((int(mg.numel()), 4)),
                reg_targets,
                strides[aidx],
                x_shifts=x_shifts[0][aidx],
                y_shifts=y_shifts[0][aidx],
            )
        num_fg = float(fg_flat.sum().item())  # host sync 3 (the loss divides by a Python float, as before)
        return cls_targets, reg_targets, obj_targets, fg_flat, l1_targets, num_fg, num_gts

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

        fg_mask, geometry_relation = self.get_geometry_constraint(
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

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.amp.autocast("cuda", enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
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
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
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

    def visualize_assign_result(self, xin, labels=None, imgs=None, save_prefix="assign_vis_"):
        # original forward logic
        outputs, x_shifts, y_shifts, expanded_strides = [], [], [], []
        # TODO: use forward logic here.

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

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, xin[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.full((1, grid.shape[1]), stride_this_level).type_as(xin[0])
            )
            outputs.append(output)

        outputs = torch.cat(outputs, 1)
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        for batch_idx, (img, num_gt, label) in enumerate(zip(imgs, nlabel, labels)):
            img = imgs[batch_idx].permute(1, 2, 0).to(torch.uint8)
            num_gt = int(num_gt)
            if num_gt == 0:
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = label[:num_gt, 1:5]
                gt_classes = label[:num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                _, fg_mask, _, matched_gt_inds, _ = self.get_assignments(  # noqa
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, expanded_strides, x_shifts,
                    y_shifts, cls_preds, obj_preds,
                )

            img = img.cpu().numpy().copy()  # copy is crucial here
            coords = torch.stack([
                ((x_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
                ((y_shifts + 0.5) * expanded_strides).flatten()[fg_mask],
            ], 1)

            xyxy_boxes = cxcywh2xyxy(gt_bboxes_per_image)
            save_name = save_prefix + str(batch_idx) + ".png"
            img = visualize_assign(img, xyxy_boxes, coords, matched_gt_inds, save_name)
            logger.info(f"save img to {save_name}")
