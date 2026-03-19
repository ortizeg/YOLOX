"""Tests for DINOXHead: RTMDet-style soft label assignment and QFL."""

import pytest
import torch

from yolox.models.dinox_head import DINOXHead
from yolox.models.losses import QualityFocalLoss


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def head() -> DINOXHead:
    """DINOXHead configured for dinox-s (width=0.50, 80 classes)."""
    return DINOXHead(
        num_classes=80, width=0.50, strides=[8, 16, 32],
        in_channels=[256, 512, 1024], act="silu", depthwise=False,
    )


@pytest.fixture
def features() -> list[torch.Tensor]:
    """Fake PAFPN outputs for dinox-s at 640x640 input."""
    return [
        torch.randn(2, 128, 80, 80),
        torch.randn(2, 256, 40, 40),
        torch.randn(2, 512, 20, 20),
    ]


@pytest.fixture
def labels() -> torch.Tensor:
    """Fake labels with shape (B, max_labels, 5). First few rows filled."""
    labs = torch.zeros(2, 50, 5)
    # image 0: 3 objects
    labs[0, 0] = torch.tensor([0, 300, 300, 50, 50])
    labs[0, 1] = torch.tensor([1, 100, 200, 40, 60])
    labs[0, 2] = torch.tensor([2, 400, 400, 80, 80])
    # image 1: 2 objects
    labs[1, 0] = torch.tensor([5, 320, 320, 60, 60])
    labs[1, 1] = torch.tensor([10, 150, 150, 30, 30])
    return labs


@pytest.fixture
def imgs() -> torch.Tensor:
    """Fake input images (B, 3, 640, 640)."""
    return torch.randn(2, 3, 640, 640)


# ------------------------------------------------------------------ #
# 1. QualityFocalLoss unit tests
# ------------------------------------------------------------------ #

class TestQualityFocalLoss:
    """Verify QFL behavior matches expected RTMDet formulation."""

    def test_negative_targets_produce_focal_loss(self) -> None:
        """When target=0, QFL = BCE(pred, 0) * sigmoid(pred)^beta."""
        qfl = QualityFocalLoss(beta=2.0, reduction="none")
        pred = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
        target = torch.zeros(1, 3)

        loss = qfl(pred, target)
        assert loss.shape == (1, 3)

        # Manual check: BCE(pred, 0) * sigmoid(pred)^2
        pred_sig = pred.detach().sigmoid()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            pred.detach(), target, reduction="none",
        )
        expected = bce * pred_sig.pow(2.0)
        torch.testing.assert_close(loss.detach(), expected, atol=1e-5, rtol=1e-5)

    def test_perfect_prediction_has_zero_loss(self) -> None:
        """When sigmoid(pred) == target exactly, loss should be near zero."""
        qfl = QualityFocalLoss(beta=2.0, reduction="none")
        # target = 0.7, so pred should be logit(0.7) = log(0.7/0.3)
        target_val = 0.7
        logit_val = torch.log(torch.tensor(target_val / (1 - target_val)))
        pred = logit_val.unsqueeze(0).unsqueeze(0)
        target = torch.tensor([[target_val]])

        loss = qfl(pred, target)
        # scale_factor = |0.7 - 0.7|^2 = 0, so loss = 0
        assert loss.item() < 1e-6

    def test_gradient_flows(self) -> None:
        """Verify gradients propagate through QFL."""
        qfl = QualityFocalLoss(beta=2.0, reduction="sum")
        pred = torch.randn(4, 3, requires_grad=True)
        target = torch.rand(4, 3) * 0.8  # soft targets in [0, 0.8]

        loss = qfl(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_reduction_modes(self) -> None:
        """Verify sum and mean reductions work correctly."""
        pred = torch.randn(4, 3)
        target = torch.rand(4, 3)

        none_loss = QualityFocalLoss(beta=2.0, reduction="none")(pred, target)
        sum_loss = QualityFocalLoss(beta=2.0, reduction="sum")(pred, target)
        mean_loss = QualityFocalLoss(beta=2.0, reduction="mean")(pred, target)

        torch.testing.assert_close(none_loss.sum(), sum_loss, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(none_loss.mean(), mean_loss, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ #
# 2. Inference output shape (same architecture as YOLOXHead)
# ------------------------------------------------------------------ #

class TestInference:
    def test_output_shape(self, head: DINOXHead, features: list[torch.Tensor]) -> None:
        head.eval()
        with torch.no_grad():
            out = head(features)
        # total_anchors = 80*80 + 40*40 + 20*20 = 8400
        assert out.shape == (2, 8400, 85), f"Expected (2, 8400, 85), got {out.shape}"

    def test_output_values_finite(self, head: DINOXHead, features: list[torch.Tensor]) -> None:
        head.eval()
        with torch.no_grad():
            out = head(features)
        assert torch.isfinite(out).all(), "Inference output contains non-finite values"


# ------------------------------------------------------------------ #
# 3. Training returns tuple of 6 values, all finite
# ------------------------------------------------------------------ #

class TestTrainingBasic:
    def test_returns_six_values(
        self, head: DINOXHead, features: list[torch.Tensor],
        labels: torch.Tensor, imgs: torch.Tensor,
    ) -> None:
        head.train()
        result = head(features, labels, imgs)
        assert isinstance(result, (tuple, dict))
        loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg = result
        for name, val in [("loss", loss), ("iou_loss", iou_loss),
                          ("obj_loss", obj_loss), ("cls_loss", cls_loss),
                          ("l1_loss", l1_loss), ("num_fg", num_fg)]:
            assert torch.is_tensor(val) or isinstance(val, (float, int)), \
                f"{name} is not a tensor or scalar"

    def test_all_finite(
        self, head: DINOXHead, features: list[torch.Tensor],
        labels: torch.Tensor, imgs: torch.Tensor,
    ) -> None:
        head.train()
        result = head(features, labels, imgs)
        loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg = result
        for name, val in [("loss", loss), ("iou_loss", iou_loss),
                          ("obj_loss", obj_loss), ("cls_loss", cls_loss),
                          ("l1_loss", l1_loss)]:
            t = val if torch.is_tensor(val) else torch.tensor(val)
            assert torch.isfinite(t).all(), f"{name} is not finite"


# ------------------------------------------------------------------ #
# 4. Loss is a scalar tensor with grad_fn
# ------------------------------------------------------------------ #

class TestLossGrad:
    def test_loss_has_grad_fn(
        self, head: DINOXHead, features: list[torch.Tensor],
        labels: torch.Tensor, imgs: torch.Tensor,
    ) -> None:
        head.train()
        loss, *_ = head(features, labels, imgs)
        assert loss.requires_grad, "Loss should require grad"
        assert loss.grad_fn is not None, "Loss should have a grad_fn"
        assert loss.dim() == 0, "Loss should be a scalar (0-dim tensor)"

    def test_backward_runs(
        self, head: DINOXHead, features: list[torch.Tensor],
        labels: torch.Tensor, imgs: torch.Tensor,
    ) -> None:
        head.train()
        loss, *_ = head(features, labels, imgs)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters() if p.requires_grad
        )
        assert has_grad, "No parameter received a gradient after backward"


# ------------------------------------------------------------------ #
# 5. Zero ground-truth objects
# ------------------------------------------------------------------ #

class TestZeroGT:
    def test_no_crash_on_zero_gt(
        self, head: DINOXHead, features: list[torch.Tensor], imgs: torch.Tensor,
    ) -> None:
        head.train()
        empty_labels = torch.zeros(2, 50, 5)
        result = head(features, empty_labels, imgs)
        loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg = result
        obj_val = obj_loss if torch.is_tensor(obj_loss) else torch.tensor(obj_loss)
        assert torch.isfinite(obj_val).all(), "obj_loss should be finite even with zero GT"


# ------------------------------------------------------------------ #
# 6. Soft center prior produces correct shape and values
# ------------------------------------------------------------------ #

class TestSoftCenterPrior:
    def test_soft_prior_shape(self, head: DINOXHead) -> None:
        """Verify soft center prior returns correct shapes."""
        gt_bboxes = torch.tensor([
            [300.0, 300.0, 40.0, 40.0],
            [100.0, 200.0, 40.0, 60.0],
        ])
        expanded_strides = torch.cat([
            torch.full((6400,), 8.0),
            torch.full((1600,), 16.0),
            torch.full((400,), 32.0),
        ]).unsqueeze(0)

        shifts_80 = torch.arange(80).float()
        x80, y80 = shifts_80.repeat(80), shifts_80.repeat_interleave(80)
        shifts_40 = torch.arange(40).float()
        x40, y40 = shifts_40.repeat(40), shifts_40.repeat_interleave(40)
        shifts_20 = torch.arange(20).float()
        x20, y20 = shifts_20.repeat(20), shifts_20.repeat_interleave(20)

        x_shifts = torch.cat([x80, x40, x20]).unsqueeze(0)
        y_shifts = torch.cat([y80, y40, y20]).unsqueeze(0)

        anchor_filter, soft_prior = head.get_geometry_constraint(
            gt_bboxes, expanded_strides, x_shifts, y_shifts,
        )
        num_filtered = anchor_filter.sum().item()
        assert soft_prior.shape == (2, num_filtered)
        assert torch.isfinite(soft_prior).all()

    def test_closer_anchors_have_lower_cost(self, head: DINOXHead) -> None:
        """Anchors closer to GT center should have lower prior cost."""
        # GT box at (300, 300) with size 200x200 -> xyxy [200, 200, 400, 400]
        gt_bboxes = torch.tensor([[300.0, 300.0, 200.0, 200.0]])
        expanded_strides = torch.full((1, 3), 8.0)
        # Three anchors all inside the box but at different distances from center:
        # Pixel coords: (252, 252), (300, 300), (372, 372) — all inside [200, 400]
        x_shifts = torch.tensor([[31.0, 37.0, 46.0]])
        y_shifts = torch.tensor([[31.0, 37.0, 46.0]])

        anchor_filter, soft_prior = head.get_geometry_constraint(
            gt_bboxes, expanded_strides, x_shifts, y_shifts,
        )
        # All 3 should be inside the GT box
        assert anchor_filter.sum() == 3, "All anchors should be inside GT box"
        # Anchor at (300,300) is closest to center, should have lowest cost
        assert soft_prior[0, 1] < soft_prior[0, 0], \
            "Center anchor should have lower cost than off-center"
        assert soft_prior[0, 1] < soft_prior[0, 2], \
            "Center anchor should have lower cost than far anchor"

    def test_outside_gt_box_filtered(self, head: DINOXHead) -> None:
        """Anchors outside all GT boxes should be filtered out."""
        gt_bboxes = torch.tensor([[160.0, 160.0, 40.0, 40.0]])  # xyxy [140, 140, 180, 180]
        expanded_strides = torch.full((1, 3), 8.0)
        # Anchor at (156, 156) inside, (164, 164) inside, (204, 204) outside
        x_shifts = torch.tensor([[19.0, 20.0, 25.0]])
        y_shifts = torch.tensor([[19.0, 20.0, 25.0]])

        anchor_filter, soft_prior = head.get_geometry_constraint(
            gt_bboxes, expanded_strides, x_shifts, y_shifts,
        )
        # Only 2 of 3 anchors should pass (the one at 204 is outside)
        assert anchor_filter.sum() == 2, \
            f"Expected 2 anchors inside GT box, got {anchor_filter.sum()}"


# ------------------------------------------------------------------ #
# 7. Soft classification cost uses IoU-weighted targets
# ------------------------------------------------------------------ #

class TestSoftClsCost:
    def test_assignment_runs(
        self, head: DINOXHead, features: list[torch.Tensor], imgs: torch.Tensor,
    ) -> None:
        """Verify get_assignments returns expected outputs without error."""
        head.train()
        num_gt = 3
        gt_bboxes = torch.tensor([
            [300.0, 300.0, 40.0, 40.0],
            [100.0, 200.0, 40.0, 60.0],
            [400.0, 400.0, 80.0, 80.0],
        ])
        gt_classes = torch.tensor([0.0, 1.0, 2.0])

        total = 8400
        bbox_preds = torch.randn(total, 4).abs() * 100 + 10
        cls_preds = torch.randn(1, total, 80)
        obj_preds = torch.randn(1, total, 1)
        expanded_strides = torch.cat([
            torch.full((6400,), 8.0),
            torch.full((1600,), 16.0),
            torch.full((400,), 32.0),
        ]).unsqueeze(0)

        shifts_80 = torch.arange(80).float()
        x80, y80 = shifts_80.repeat(80), shifts_80.repeat_interleave(80)
        shifts_40 = torch.arange(40).float()
        x40, y40 = shifts_40.repeat(40), shifts_40.repeat_interleave(40)
        shifts_20 = torch.arange(20).float()
        x20, y20 = shifts_20.repeat(20), shifts_20.repeat_interleave(20)

        x_shifts = torch.cat([x80, x40, x20]).unsqueeze(0)
        y_shifts = torch.cat([y80, y40, y20]).unsqueeze(0)

        result = head.get_assignments(
            0, num_gt, gt_bboxes, gt_classes, bbox_preds,
            expanded_strides, x_shifts, y_shifts, cls_preds, obj_preds,
        )
        assert len(result) == 5, "get_assignments should return 5 values"
        gt_matched_classes, fg_mask, pred_ious, matched_gt_inds, num_fg = result
        assert num_fg > 0, "Should have at least 1 foreground match"
        assert fg_mask.sum() == num_fg


# ------------------------------------------------------------------ #
# 8. DINOXHead has same parameter count as YOLOXHead
# ------------------------------------------------------------------ #

class TestArchitectureCompat:
    def test_same_param_count_as_yolox(self) -> None:
        """DINOXHead architecture is identical to YOLOXHead (same conv layers)."""
        from yolox.models.yolo_head import YOLOXHead

        yolox = YOLOXHead(num_classes=80, width=0.50)
        dinox = DINOXHead(num_classes=80, width=0.50)

        yolox_params = sum(p.numel() for p in yolox.parameters())
        dinox_params = sum(p.numel() for p in dinox.parameters())
        assert yolox_params == dinox_params, \
            f"YOLOX ({yolox_params}) and DINOX ({dinox_params}) should have same param count"

    def test_depthwise_variant(self) -> None:
        """Depthwise DINOXHead should work and have fewer params."""
        regular = DINOXHead(num_classes=80, width=0.50, depthwise=False)
        dw = DINOXHead(num_classes=80, width=0.50, depthwise=True)
        regular_params = sum(p.numel() for p in regular.parameters())
        dw_params = sum(p.numel() for p in dw.parameters())
        assert dw_params < regular_params


# ------------------------------------------------------------------ #
# 9. Experiment config loads correctly
# ------------------------------------------------------------------ #

class TestExpConfig:
    def test_dinox_s_exp_loads(self) -> None:
        """Verify dinox_s experiment can be loaded by name."""
        from yolox.exp import get_exp

        exp = get_exp(exp_name="dinox_s")
        model = exp.get_model()
        assert hasattr(model, "head")
        assert isinstance(model.head, DINOXHead)

    def test_dinox_s_model_forward(self) -> None:
        """Verify dinox_s model runs inference."""
        from yolox.exp import get_exp

        exp = get_exp(exp_name="dinox_s")
        model = exp.get_model()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 640, 640))
        assert out.shape == (1, 8400, 85)
