#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# DINOv2 feature distillation for YOLOX (Phase 4 v4).
# Teacher is frozen DINOv2-B/14, removed at inference.
#
# Additive loss (ViTKD/LRRA-style):
#   L_total = L_detection + λ_feat * L_feature + λ_cls * L_cls_token + λ_attn * L_attention
#
# Three distillation signals:
# 1. Feature alignment: dark5 → DINOv2 layer 12 (single deepest level)
# 2. CLS token: GAP(dark5) → DINOv2 [CLS] token (global context)
# 3. Attention maps: spatial attention distillation via KL divergence

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger


class StudentProjector(nn.Module):
    """2-layer projector for student features.

    Architecture: Conv1x1 → BN → GELU → Conv1x1
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
        return self.proj(x)


class CLSProjector(nn.Module):
    """Projects GAP'd student features to match CLS token dimension.

    Architecture: Linear → BN1d → GELU → Linear
    """

    def __init__(self, student_channels: int, teacher_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_channels, teacher_dim, bias=False),
            nn.BatchNorm1d(teacher_dim),
            nn.GELU(),
            nn.Linear(teacher_dim, teacher_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class YOLOXDistill(nn.Module):
    """YOLOX with DINOv2 feature distillation during training.

    Additive loss formulation (following ViTKD/LRRA best practices for
    ViT-to-CNN distillation):

        L_total = L_detection + λ_feat * L_feature + λ_cls * L_cls + λ_attn * L_attn

    Three distillation signals address different aspects of the ViT→CNN gap:

    1. **Feature alignment** (dark5 ↔ layer 12 only): Single deepest level
       where CNN and ViT have the most semantic overlap. Avoids negative
       transfer from misaligned early layers.

    2. **CLS token** (GAP(dark5) ↔ [CLS]): Transfers global scene context
       that CNNs lack due to local inductive bias.

    3. **Attention maps** (spatial attention KL divergence): Transfers
       "where to look" rather than "exact feature values" — architecture-
       agnostic signal that works across CNN/ViT gap.

    At inference, all distillation components are removed — zero overhead.

    Args:
        model: Base YOLOX model.
        teacher_model: DINOv2 model name.
        lambda_feat: Feature alignment loss weight. Default 1.0.
        lambda_cls: CLS token loss weight. Default 1.0.
        lambda_attn: Attention map loss weight. Default 0.5.
        teacher_layer: Which ViT layer to distill from. Default 12 (last).
        teacher_precision: Precision for frozen teacher.
        student_channels: Channel count for dark5 (512 for YOLOX-S width=0.5).
    """

    def __init__(
        self,
        model: nn.Module,
        teacher_model: str = "dinov2_vitb14",
        lambda_feat: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_attn: float = 0.5,
        teacher_layer: int = 12,
        teacher_precision: str = "float16",
        student_channels: int = 512,
    ) -> None:
        super().__init__()
        self.model = model
        self.lambda_feat = lambda_feat
        self.lambda_cls = lambda_cls
        self.lambda_attn = lambda_attn
        self.teacher_layer = teacher_layer
        self.teacher_precision = teacher_precision

        # Teacher loaded lazily (plain attr, excluded from DDP/EMA)
        self._teacher_model_name = teacher_model
        self._teacher = None
        self._teacher_loaded = False

        teacher_dim = 768  # ViT-B embed_dim
        self.patch_size = 14
        self._num_heads = 12  # ViT-B/14 has 12 attention heads

        # Feature projector: dark5 → teacher dim (single level)
        self.feat_projector = StudentProjector(student_channels, teacher_dim)

        # CLS token projector: GAP(dark5) → teacher dim
        self.cls_projector = CLSProjector(student_channels, teacher_dim)

        logger.info(
            "Distillation v4: additive loss, λ_feat={}, λ_cls={}, λ_attn={}, "
            "teacher_layer={}, student_ch={}",
            lambda_feat, lambda_cls, lambda_attn, teacher_layer, student_channels,
        )

    def _ensure_teacher_loaded(self, device: torch.device) -> None:
        if self._teacher_loaded:
            return
        logger.info("Loading DINOv2 teacher: {} on {}", self._teacher_model_name, device)
        self._teacher = torch.hub.load(
            "facebookresearch/dinov2", self._teacher_model_name, pretrained=True,
        )
        self._teacher.to(device).eval()
        for param in self._teacher.parameters():
            param.requires_grad = False
        self._teacher_loaded = True
        logger.info("DINOv2 teacher loaded: {:,} params (frozen)",
                     sum(p.numel() for p in self._teacher.parameters()))

    def forward(self, x: torch.Tensor, targets=None):
        if self.training:
            assert targets is not None
            self._ensure_teacher_loaded(x.device)

            # Extract dark5 (deepest pre-FPN backbone features)
            pafpn = self.model.backbone
            darknet_outs = pafpn.backbone(x)
            dark5 = darknet_outs[pafpn.in_features[-1]]  # stride 32, 512ch

            # Run full PAFPN + head for detection loss
            fpn_outs = pafpn(x)
            det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.model.head(
                fpn_outs, targets, x
            )

            # Get teacher outputs (features, CLS token, attention map)
            teacher_feat, teacher_cls, teacher_attn = self._get_teacher_outputs(x)

            # 1. Feature alignment loss (dark5 ↔ layer 12)
            feat_loss = self._feature_loss(dark5, teacher_feat)

            # 2. CLS token loss (GAP(dark5) ↔ [CLS])
            cls_token_loss = self._cls_token_loss(dark5, teacher_cls)

            # 3. Attention map loss (spatial attention KL divergence)
            attn_loss = self._attention_loss(dark5, teacher_attn)

            # Additive loss (ViTKD-style)
            total_loss = (
                det_loss
                + self.lambda_feat * feat_loss
                + self.lambda_cls * cls_token_loss
                + self.lambda_attn * attn_loss
            )

            outputs = {
                "total_loss": total_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "distill_loss": (feat_loss + cls_token_loss + attn_loss).detach(),
            }
        else:
            outputs = self.model.head(self.model.backbone(x))

        return outputs

    @torch.no_grad()
    def _get_teacher_outputs(self, x: torch.Tensor):
        """Extract feature map, CLS token, and attention from teacher."""
        B = x.shape[0]
        dtype = torch.float16 if self.teacher_precision == "float16" else torch.float32

        # Normalize input for DINOv2
        x_rgb = x[:, [2, 1, 0], :, :]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_norm = (x_rgb / 255.0 - mean) / std

        H, W = x_norm.shape[2:]
        pH = (H // self.patch_size) * self.patch_size
        pW = (W // self.patch_size) * self.patch_size
        if pH != H or pW != W:
            x_norm = F.interpolate(x_norm, size=(pH, pW), mode="bilinear", align_corners=False)

        with torch.amp.autocast("cuda", dtype=dtype):
            x_t = self._teacher.prepare_tokens_with_masks(x_norm)

            attn_map = None
            for i, blk in enumerate(self._teacher.blocks):
                # Capture attention from the target layer
                if (i + 1) == self.teacher_layer:
                    # Extract attention weights from this block
                    # DINOv2 block.attn is the attention module
                    attn_module = blk.attn
                    B_t, N, C = x_t.shape

                    # Compute Q, K manually to get attention weights
                    qkv = attn_module.qkv(x_t)
                    qkv = qkv.reshape(B_t, N, 3, self._num_heads, C // self._num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    scale = (C // self._num_heads) ** -0.5
                    attn_weights = (q @ k.transpose(-2, -1)) * scale
                    attn_weights = attn_weights.softmax(dim=-1)
                    # Average over heads: (B, num_heads, N, N) → (B, N, N)
                    attn_map = attn_weights.mean(dim=1).float()

                x_t = blk(x_t)

                if (i + 1) == self.teacher_layer:
                    # Extract features and CLS token
                    cls_token = x_t[:, 0].float()  # (B, 768)
                    patch_tokens = x_t[:, 1:]  # (B, N_patches, 768)
                    H_t = W_t = int(patch_tokens.shape[1] ** 0.5)
                    feat_map = patch_tokens.reshape(B, H_t, W_t, -1).permute(0, 3, 1, 2).float()

        return feat_map, cls_token, attn_map

    def _feature_loss(self, dark5: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """Cosine similarity loss between dark5 and teacher spatial features."""
        # Project student to teacher dim
        student_proj = self.feat_projector(dark5)  # (B, 768, H_s, W_s)

        # Interpolate teacher to match student spatial size
        target_size = (student_proj.shape[2], student_proj.shape[3])
        teacher_resized = F.interpolate(
            teacher_feat, size=target_size, mode="bilinear", align_corners=False,
        )

        # L2 normalize both, then cosine similarity
        student_norm = F.normalize(student_proj, p=2, dim=1)
        teacher_norm = F.normalize(teacher_resized, p=2, dim=1)

        cos_sim = (student_norm * teacher_norm).sum(dim=1)  # (B, H, W)
        return (1.0 - cos_sim).mean()

    def _cls_token_loss(self, dark5: torch.Tensor, teacher_cls: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between GAP(dark5) and teacher CLS token."""
        # Global average pool student features → (B, C)
        student_gap = dark5.mean(dim=[2, 3])  # (B, 512)

        # Project to teacher dim
        student_proj = self.cls_projector(student_gap)  # (B, 768)

        # L2 normalize both
        student_norm = F.normalize(student_proj, p=2, dim=1)
        teacher_norm = F.normalize(teacher_cls, p=2, dim=1)

        cos_sim = (student_norm * teacher_norm).sum(dim=1)  # (B,)
        return (1.0 - cos_sim).mean()

    def _attention_loss(self, dark5: torch.Tensor, teacher_attn: torch.Tensor) -> torch.Tensor:
        """KL divergence on spatial attention distributions.

        Transfers "where to look" from teacher to student — architecture-
        agnostic since it compares attention patterns, not feature values.
        """
        B = dark5.shape[0]

        # Student spatial attention: channel-wise mean → softmax over spatial positions
        student_spatial = dark5.mean(dim=1)  # (B, H_s, W_s)
        student_flat = student_spatial.view(B, -1)  # (B, H_s*W_s)
        student_attn = F.log_softmax(student_flat / 0.1, dim=1)  # temperature=0.1

        # Teacher attention: use patch-to-patch attention (exclude CLS)
        # teacher_attn shape: (B, N_total, N_total) where N_total = 1 + N_patches
        # Extract patch-to-patch: mean over query patches → spatial distribution
        patch_attn = teacher_attn[:, 1:, 1:]  # (B, N_patches, N_patches)
        teacher_spatial = patch_attn.mean(dim=1)  # (B, N_patches) — avg attention received
        teacher_attn_dist = F.softmax(teacher_spatial / 0.1, dim=1)  # temperature=0.1

        # Resize teacher attention to match student spatial size
        H_s, W_s = dark5.shape[2], dark5.shape[3]
        H_t = int(teacher_spatial.shape[1] ** 0.5)
        teacher_2d = teacher_attn_dist.view(B, 1, H_t, H_t)
        teacher_resized = F.interpolate(
            teacher_2d, size=(H_s, W_s), mode="bilinear", align_corners=False,
        ).view(B, -1)
        # Re-normalize after interpolation
        teacher_resized = teacher_resized / teacher_resized.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # KL divergence: KL(teacher || student)
        kl_loss = F.kl_div(student_attn, teacher_resized, reduction="batchmean")

        return kl_loss

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        self.model.visualize(x, targets, save_prefix)
