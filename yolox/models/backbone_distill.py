#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""DINOv2 → CSPDarknet backbone distillation on ImageNet.

Trains CSPDarknet to match DINOv2 features while also learning ImageNet
classification. The distilled backbone can then be used as a drop-in
replacement for YOLOX detection training.

Combined loss:
  L = L_cls + λ_feat * L_feature + λ_cls * L_cls_token + λ_rkd * L_relational

Three distillation signals:
1. Feature alignment: dark5 → DINOv2 layer 12 (cosine similarity)
2. CLS token: GAP(dark5) → DINOv2 [CLS] token (cosine similarity)
3. Relational KD: match pairwise similarity structure between samples
   (architecture-agnostic, captures "how samples relate" not raw features)

Best with input_size=448: dark5 is 14x14, closely matching DINOv2's
patch grid (448/14=32 patches per side). This spatial alignment is
critical for feature distillation quality.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger


class BackboneDistillModel(nn.Module):
    """CSPDarknet backbone with DINOv2 teacher for distillation.

    Args:
        backbone: CSPDarknet backbone instance.
        num_classes: Number of ImageNet classes (1000).
        teacher_model: DINOv2 model name for torch.hub.
        teacher_layer: Which ViT layer to distill from.
        teacher_dim: Teacher embedding dimension (768 for ViT-B).
        student_dim: Student dark5 channel count (512 for YOLOX-S).
        lambda_feat: Feature alignment loss weight.
        lambda_cls_token: CLS token loss weight.
        lambda_rkd: Relational knowledge distillation weight.
        teacher_precision: Precision for frozen teacher.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int = 1000,
        teacher_model: str = "dinov2_vitb14",
        teacher_layer: int = 12,
        teacher_dim: int = 768,
        student_dim: int = 512,
        lambda_feat: float = 2.0,
        lambda_cls_token: float = 1.0,
        lambda_rkd: float = 1.0,
        teacher_precision: str = "float16",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.lambda_feat = lambda_feat
        self.lambda_cls_token = lambda_cls_token
        self.lambda_rkd = lambda_rkd
        self.teacher_layer = teacher_layer
        self.teacher_precision = teacher_precision
        self.patch_size = 14
        self._num_heads = 12

        # Classification head (auxiliary task)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(student_dim, num_classes),
        )

        # Feature projector: dark5 → teacher dim
        self.feat_projector = nn.Sequential(
            nn.Conv2d(student_dim, teacher_dim, 1, bias=False),
            nn.BatchNorm2d(teacher_dim),
            nn.GELU(),
            nn.Conv2d(teacher_dim, teacher_dim, 1, bias=False),
        )

        # CLS token projector: GAP(dark5) → teacher dim
        self.cls_projector = nn.Sequential(
            nn.Linear(student_dim, teacher_dim, bias=False),
            nn.BatchNorm1d(teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, teacher_dim, bias=False),
        )

        # Teacher loaded lazily
        self._teacher_model_name = teacher_model
        self._teacher = None
        self._teacher_loaded = False

        logger.info(
            "Backbone distillation: λ_feat={}, λ_cls={}, λ_rkd={}, "
            "teacher_layer={}, student_dim={}, teacher_dim={}",
            lambda_feat, lambda_cls_token, lambda_rkd,
            teacher_layer, student_dim, teacher_dim,
        )

    def _ensure_teacher_loaded(self, device: torch.device) -> None:
        if self._teacher_loaded:
            return
        logger.info("Loading DINOv2 teacher: {}", self._teacher_model_name)
        self._teacher = torch.hub.load(
            "facebookresearch/dinov2", self._teacher_model_name, pretrained=True,
        )
        self._teacher.to(device).eval()
        for param in self._teacher.parameters():
            param.requires_grad = False
        self._teacher_loaded = True
        logger.info("DINOv2 teacher loaded: {:,} params (frozen)",
                     sum(p.numel() for p in self._teacher.parameters()))

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        # Run student backbone
        backbone_outs = self.backbone(x)
        dark5 = backbone_outs["dark5"]  # (B, 512, H/32, W/32)

        if not self.training:
            return self.cls_head(dark5)

        assert targets is not None

        # Classification loss
        logits = self.cls_head(dark5)
        cls_loss = F.cross_entropy(logits, targets)

        # Get teacher features
        self._ensure_teacher_loaded(x.device)
        teacher_feat, teacher_cls = self._get_teacher_outputs(x)

        # Feature alignment loss (cosine similarity)
        feat_loss = self._feature_loss(dark5, teacher_feat)

        # CLS token loss (cosine similarity)
        cls_token_loss = self._cls_token_loss(dark5, teacher_cls)

        # Relational knowledge distillation
        rkd_loss = self._relational_loss(dark5, teacher_cls)

        # Combined loss
        total_loss = (
            cls_loss
            + self.lambda_feat * feat_loss
            + self.lambda_cls_token * cls_token_loss
            + self.lambda_rkd * rkd_loss
        )

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "feat_loss": feat_loss.detach(),
            "cls_token_loss": cls_token_loss.detach(),
            "rkd_loss": rkd_loss.detach(),
            "acc1": (logits.argmax(dim=1) == targets).float().mean().detach(),
        }

    @torch.no_grad()
    def _get_teacher_outputs(self, x: torch.Tensor):
        """Extract feature map and CLS token from teacher."""
        B = x.shape[0]
        dtype = torch.float16 if self.teacher_precision == "float16" else torch.float32

        H, W = x.shape[2:]
        pH = (H // self.patch_size) * self.patch_size
        pW = (W // self.patch_size) * self.patch_size
        x_in = x
        if pH != H or pW != W:
            x_in = F.interpolate(x, size=(pH, pW), mode="bilinear", align_corners=False)

        with torch.amp.autocast("cuda", dtype=dtype):
            x_t = self._teacher.prepare_tokens_with_masks(x_in)

            for i, blk in enumerate(self._teacher.blocks):
                x_t = blk(x_t)
                if (i + 1) == self.teacher_layer:
                    cls_token = x_t[:, 0].float()  # (B, 768)
                    patch_tokens = x_t[:, 1:]  # (B, N_patches, 768)
                    H_t = W_t = int(patch_tokens.shape[1] ** 0.5)
                    feat_map = patch_tokens.reshape(B, H_t, W_t, -1).permute(0, 3, 1, 2).float()
                    break

        return feat_map, cls_token

    def _feature_loss(self, dark5: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Cosine similarity loss between dark5 and teacher features."""
        student_proj = self.feat_projector(dark5)

        target_size = (student_proj.shape[2], student_proj.shape[3])
        teacher_resized = F.interpolate(
            teacher_feat, size=target_size, mode="bilinear", align_corners=False,
        )

        student_norm = F.normalize(student_proj, p=2, dim=1)
        teacher_norm = F.normalize(teacher_resized, p=2, dim=1)

        cos_sim = (student_norm * teacher_norm).sum(dim=1)
        return (1.0 - cos_sim).mean()

    def _cls_token_loss(self, dark5: torch.Tensor, teacher_cls: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between GAP(dark5) and teacher CLS token."""
        student_gap = dark5.mean(dim=[2, 3])
        student_proj = self.cls_projector(student_gap)

        student_norm = F.normalize(student_proj, p=2, dim=1)
        teacher_norm = F.normalize(teacher_cls, p=2, dim=1)

        cos_sim = (student_norm * teacher_norm).sum(dim=1)
        return (1.0 - cos_sim).mean()

    def _relational_loss(self, dark5: torch.Tensor, teacher_cls: torch.Tensor) -> torch.Tensor:
        """Relational Knowledge Distillation (RKD distance + angle).

        Matches pairwise distance and angle structures between samples.
        Architecture-agnostic: compares relationships, not raw features.
        Based on "Relational Knowledge Distillation" (Park et al., CVPR 2019).
        """
        # Student: GAP(dark5) as sample embedding
        student_emb = dark5.mean(dim=[2, 3])  # (B, 512)
        student_emb = F.normalize(student_emb, p=2, dim=1)

        # Teacher: CLS token as sample embedding
        teacher_emb = F.normalize(teacher_cls, p=2, dim=1)  # (B, 768)

        # --- Distance-wise RKD ---
        # Pairwise distances (B, B)
        s_dist = torch.cdist(student_emb, student_emb, p=2)
        t_dist = torch.cdist(teacher_emb, teacher_emb, p=2)

        # Normalize by mean distance (makes it scale-invariant)
        s_dist = s_dist / (s_dist.mean() + 1e-8)
        t_dist = t_dist / (t_dist.mean() + 1e-8)

        # Huber loss on pairwise distances
        dist_loss = F.smooth_l1_loss(s_dist, t_dist)

        # --- Angle-wise RKD ---
        # For every triplet (i, j, k), match the angle at j
        # Efficient: compute all pairwise difference vectors, then cosine
        B = student_emb.shape[0]
        if B < 3:
            return dist_loss

        # Difference vectors: (B, B, D)
        s_diff = student_emb.unsqueeze(0) - student_emb.unsqueeze(1)
        t_diff = teacher_emb.unsqueeze(0) - teacher_emb.unsqueeze(1)

        # Cosine of angles between difference vectors at each anchor
        # For each pair of difference vectors from the same anchor point
        s_diff_norm = F.normalize(s_diff, p=2, dim=2)
        t_diff_norm = F.normalize(t_diff, p=2, dim=2)

        # Angle matrix: cosine similarity between all pairs of directions
        # (B, B, B) would be too large — use a sampled version
        # Instead, match the pairwise cosine similarity matrix
        s_angle = torch.bmm(s_diff_norm, s_diff_norm.transpose(1, 2))  # (B, B, B)
        t_angle = torch.bmm(t_diff_norm, t_diff_norm.transpose(1, 2))  # (B, B, B)

        angle_loss = F.smooth_l1_loss(s_angle, t_angle)

        return dist_loss + angle_loss

    def get_backbone_state_dict(self) -> dict:
        """Extract only the backbone weights for downstream use."""
        return {k: v for k, v in self.backbone.state_dict().items()}
