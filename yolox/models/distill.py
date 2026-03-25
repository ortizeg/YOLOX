#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# DINOv2 feature distillation for YOLOX (Phase 4).
# Teacher is frozen DINOv2-B/14, removed at inference.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger


class DistillProjector(nn.Module):
    """Projects student features to match teacher embedding dimension."""

    def __init__(self, student_channels: int, teacher_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_channels, teacher_dim, 1, bias=False),
            nn.BatchNorm2d(teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class YOLOXDistill(nn.Module):
    """YOLOX with DINOv2 feature distillation during training.

    Wraps an existing YOLOX model and adds a frozen DINOv2-B/14 teacher.
    During training, RAW BACKBONE features (pre-FPN) are aligned to
    teacher features via cosine similarity loss through lightweight
    projector modules. At inference, the teacher and projectors are
    completely removed — zero overhead.

    Feature alignment (pre-FPN backbone stages → teacher layers):
    - Student dark3 (stride 8) ↔ Teacher layer 4
    - Student dark4 (stride 16) ↔ Teacher layer 8
    - Student dark5 (stride 32) ↔ Teacher layer 12

    Args:
        model: Base YOLOX model (backbone + head).
        teacher_model: DINOv2 model name for torch.hub.
        distill_weight: Weight for total distillation loss.
        distill_levels: ViT layer indices to extract features from.
        teacher_precision: Precision for frozen teacher.
        student_channels: Channel counts for raw backbone stages
            (dark3, dark4, dark5 BEFORE FPN).
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_model: str = "dinov2_vitb14",
        distill_weight: float = 0.5,
        distill_levels: list[int] = [4, 8, 12],
        teacher_precision: str = "float16",
        student_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.distill_weight = distill_weight
        self.distill_levels = distill_levels
        self.teacher_precision = teacher_precision

        # Load frozen DINOv2 teacher
        logger.info("Loading DINOv2 teacher: {}", teacher_model)
        self.teacher = torch.hub.load(
            "facebookresearch/dinov2", teacher_model, pretrained=True,
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        teacher_dim = self.teacher.embed_dim  # 768 for ViT-B
        self.patch_size = self.teacher.patch_size  # 14

        # Projectors align raw backbone features (pre-FPN) to teacher dim
        if student_channels is None:
            # YOLOX-S raw CSPDarknet: dark3=128, dark4=256, dark5=512 (width=0.5)
            student_channels = [128, 256, 512]

        self.projectors = nn.ModuleList([
            DistillProjector(ch, teacher_dim) for ch in student_channels
        ])

        logger.info(
            "Distillation: {} levels, weight={}, teacher_dim={}, "
            "student_channels={} (pre-FPN), loss=cosine",
            len(distill_levels), distill_weight, teacher_dim, student_channels,
        )

    def forward(self, x: torch.Tensor, targets=None):
        if self.training:
            assert targets is not None

            # Extract raw backbone features (pre-FPN) for distillation
            pafpn = self.model.backbone
            darknet_outs = pafpn.backbone(x)
            raw_backbone_feats = [darknet_outs[f] for f in pafpn.in_features]
            # raw_backbone_feats = [dark3, dark4, dark5] — pre-FPN

            # Run full PAFPN for detection (recomputes CSPDarknet but keeps it simple)
            fpn_outs = pafpn(x)
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.model.head(
                fpn_outs, targets, x
            )

            # Compute distillation on raw backbone features
            distill_loss = self._compute_distill_loss(x, raw_backbone_feats)

            total_loss = loss + self.distill_weight * distill_loss

            outputs = {
                "total_loss": total_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "distill_loss": distill_loss.detach(),
            }
        else:
            # Inference: just run the base model, no teacher
            outputs = self.model.head(self.model.backbone(x))

        return outputs

    @torch.no_grad()
    def _get_teacher_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate features from frozen DINOv2 teacher."""
        B = x.shape[0]
        dtype = torch.float16 if self.teacher_precision == "float16" else torch.float32

        # DINOv2 expects ImageNet-normalized RGB input
        # YOLOX preprocesses to [0, 255] BGR
        x_rgb = x[:, [2, 1, 0], :, :]  # BGR -> RGB
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x_rgb / 255.0 - mean) / std

        # Resize to multiple of patch_size
        H, W = x_norm.shape[2:]
        pH = (H // self.patch_size) * self.patch_size
        pW = (W // self.patch_size) * self.patch_size
        if pH != H or pW != W:
            x_norm = F.interpolate(x_norm, size=(pH, pW), mode="bilinear", align_corners=False)

        with torch.amp.autocast("cuda", dtype=dtype):
            x_teacher = self.teacher.prepare_tokens_with_masks(x_norm)

            features = []
            for i, blk in enumerate(self.teacher.blocks):
                x_teacher = blk(x_teacher)
                if (i + 1) in self.distill_levels:
                    patch_tokens = x_teacher[:, 1:]  # Remove CLS token
                    H = W = int(patch_tokens.shape[1] ** 0.5)
                    feat = patch_tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)
                    features.append(feat.float())

        return features

    def _compute_distill_loss(
        self, x: torch.Tensor, backbone_feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute cosine similarity distillation loss on pre-FPN features."""
        teacher_feats = self._get_teacher_features(x)

        total_loss = torch.tensor(0.0, device=x.device)
        for student_feat, teacher_feat, projector in zip(
            backbone_feats, teacher_feats, self.projectors
        ):
            # Project student to teacher dimension
            projected = projector(student_feat)  # (B, 768, H_s, W_s)

            # Interpolate teacher to match student spatial size
            teacher_resized = F.interpolate(
                teacher_feat,
                size=projected.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            # Cosine similarity loss: 1 - cos_sim (per spatial position, averaged)
            # Reshape to (B*H*W, 768) for cosine computation
            B, C, H, W = projected.shape
            proj_flat = projected.permute(0, 2, 3, 1).reshape(-1, C)
            teach_flat = teacher_resized.permute(0, 2, 3, 1).reshape(-1, C)
            cos_sim = F.cosine_similarity(proj_flat, teach_flat, dim=1)
            total_loss = total_loss + (1.0 - cos_sim).mean()

        return total_loss / len(teacher_feats)

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        self.model.visualize(x, targets, save_prefix)
