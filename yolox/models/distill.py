#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# DINOv2 feature distillation for YOLOX (Phase 4).
# Teacher is frozen DINOv2-B/14, removed at inference.
#
# Uses interpolated loss (Hinton et al.):
#   L_total = α * L_detection + (1 - α) * L_distill
# with α=0.3 (teacher-focused, 70% distillation weight).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger


class TeacherAdapter(nn.Module):
    """Adapts single-scale ViT features to a target spatial resolution.

    Learns to refine bilinearly-interpolated ViT features into
    representations that are more useful for the student to learn from.

    Architecture: interpolate → Conv3x3 → BN → GELU → Conv1x1 → BN
    """

    def __init__(self, teacher_dim: int, target_size: tuple[int, int]) -> None:
        super().__init__()
        self.target_size = target_size
        self.adapter = nn.Sequential(
            nn.Conv2d(teacher_dim, teacher_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(teacher_dim),
            nn.GELU(),
            nn.Conv2d(teacher_dim, teacher_dim, 1, bias=False),
            nn.BatchNorm2d(teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        return self.adapter(x)


class StudentProjector(nn.Module):
    """2-layer projector for student features with L2 normalization.

    Architecture: Conv1x1 → BN → GELU → Conv1x1 → L2_norm
    """

    def __init__(self, student_channels: int, teacher_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_channels, teacher_dim, 1, bias=False),
            nn.BatchNorm2d(teacher_dim),
            nn.GELU(),
            nn.Conv2d(teacher_dim, teacher_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # L2 normalize channel-wise at each spatial position
        return F.normalize(x, p=2, dim=1)


class YOLOXDistill(nn.Module):
    """YOLOX with DINOv2 feature distillation during training.

    Uses interpolated loss following knowledge distillation best practices
    (Hinton et al. 2015):

        L_total = α * L_detection + (1 - α) * L_distill

    where α controls the balance (lower α = more teacher influence).

    Feature alignment uses pre-FPN backbone features (dark3/4/5) aligned
    to multi-scale teacher features via L2-normalized cosine similarity.
    Teacher features are refined through learned spatial adapters.

    At inference, the teacher, adapters, and projectors are completely
    removed — zero overhead.

    Args:
        model: Base YOLOX model.
        teacher_model: DINOv2 model name for torch.hub.
        distill_alpha: Interpolation weight for detection loss. Default 0.3
            (70% distillation, 30% detection — teacher-focused).
        distill_levels: ViT layer indices to extract features from.
        teacher_precision: Precision for frozen teacher.
        student_channels: Channel counts for raw backbone stages.
        target_sizes: Spatial sizes for multi-scale teacher adapters.
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_model: str = "dinov2_vitb14",
        distill_alpha: float = 0.3,
        distill_levels: list[int] = [4, 8, 12],
        teacher_precision: str = "float16",
        student_channels: list[int] | None = None,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.distill_alpha = distill_alpha
        self.distill_levels = distill_levels
        self.teacher_precision = teacher_precision

        # Load frozen DINOv2 teacher (deferred to first forward to avoid
        # DDP broadcast of 86M frozen params during model init)
        self._teacher_model_name = teacher_model
        self.teacher = None
        self._teacher_loaded = False

        # ViT-B/14 constants (known ahead of time)
        teacher_dim = 768  # embed_dim for dinov2_vitb14
        self.patch_size = 14

        if student_channels is None:
            student_channels = [128, 256, 512]

        if target_sizes is None:
            # Match CSPDarknet spatial sizes for 640x640 input
            target_sizes = [(80, 80), (40, 40), (20, 20)]

        # Teacher adapters: transform single-scale ViT → multi-scale
        self.teacher_adapters = nn.ModuleList([
            TeacherAdapter(teacher_dim, size) for size in target_sizes
        ])

        # Student projectors: 2-layer with L2 normalization
        self.projectors = nn.ModuleList([
            StudentProjector(ch, teacher_dim) for ch in student_channels
        ])

        logger.info(
            "Distillation: α={} ({}% teacher), {} levels, "
            "teacher_dim={}, targets={}, loss=normalized_cosine",
            distill_alpha, int((1 - distill_alpha) * 100),
            len(distill_levels), teacher_dim, target_sizes,
        )

    def _ensure_teacher_loaded(self, device: torch.device) -> None:
        """Lazy-load the frozen DINOv2 teacher on first use."""
        if self._teacher_loaded:
            return
        logger.info("Loading DINOv2 teacher: {} on {}", self._teacher_model_name, device)
        self.teacher = torch.hub.load(
            "facebookresearch/dinov2", self._teacher_model_name, pretrained=True,
        )
        self.teacher.to(device).eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self._teacher_loaded = True
        logger.info("DINOv2 teacher loaded: {:,} params (frozen)",
                     sum(p.numel() for p in self.teacher.parameters()))

    def forward(self, x: torch.Tensor, targets=None):
        if self.training:
            assert targets is not None
            self._ensure_teacher_loaded(x.device)

            # Extract raw backbone features (pre-FPN)
            pafpn = self.model.backbone
            darknet_outs = pafpn.backbone(x)
            raw_backbone_feats = [darknet_outs[f] for f in pafpn.in_features]

            # Run full PAFPN + head for detection loss
            fpn_outs = pafpn(x)
            det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.model.head(
                fpn_outs, targets, x
            )

            # Compute distillation loss on raw backbone features
            distill_loss = self._compute_distill_loss(x, raw_backbone_feats)

            # Interpolated loss: α * detection + (1 - α) * distillation
            total_loss = (
                self.distill_alpha * det_loss
                + (1.0 - self.distill_alpha) * distill_loss
            )

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
            outputs = self.model.head(self.model.backbone(x))

        return outputs

    @torch.no_grad()
    def _get_teacher_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate features from frozen DINOv2 teacher."""
        B = x.shape[0]
        dtype = torch.float16 if self.teacher_precision == "float16" else torch.float32

        # DINOv2 expects ImageNet-normalized RGB input
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
                    patch_tokens = x_teacher[:, 1:]
                    H_t = W_t = int(patch_tokens.shape[1] ** 0.5)
                    feat = patch_tokens.reshape(B, H_t, W_t, -1).permute(0, 3, 1, 2)
                    features.append(feat.float())

        return features

    def _compute_distill_loss(
        self, x: torch.Tensor, backbone_feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute L2-normalized cosine distillation loss."""
        teacher_feats = self._get_teacher_features(x)

        total_loss = torch.tensor(0.0, device=x.device)
        for student_feat, teacher_feat, projector, adapter in zip(
            backbone_feats, teacher_feats, self.projectors, self.teacher_adapters
        ):
            # Student: project + L2 normalize (done inside StudentProjector)
            student_proj = projector(student_feat)  # (B, 768, H, W), L2-normed

            # Teacher: adapt to target spatial size + L2 normalize
            teacher_adapted = adapter(teacher_feat)  # (B, 768, H, W)
            teacher_normed = F.normalize(teacher_adapted, p=2, dim=1)

            # Cosine similarity loss (both are L2-normalized → dot product)
            # 1 - cos_sim per spatial position, averaged
            cos_sim = (student_proj * teacher_normed).sum(dim=1)  # (B, H, W)
            total_loss = total_loss + (1.0 - cos_sim).mean()

        return total_loss / len(teacher_feats)

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        self.model.visualize(x, targets, save_prefix)
