"""Tests for Distribution Focal Loss (DFL) and Integral module."""

import pytest
import torch

from yolox.models.dinox_head import DINOXHead, _bbox2distance, _distance2bbox
from yolox.models.losses import DistributionFocalLoss, Integral


# ------------------------------------------------------------------ #
# 1. Integral module
# ------------------------------------------------------------------ #

class TestIntegral:
    def test_output_shape(self) -> None:
        """Integral converts (N, 4*(reg_max+1)) -> (N, 4)."""
        integral = Integral(reg_max=16)
        x = torch.randn(10, 4 * 17)
        out = integral(x)
        assert out.shape == (10, 4)

    def test_uniform_distribution_gives_midpoint(self) -> None:
        """Uniform logits should give reg_max/2 as the expected value."""
        integral = Integral(reg_max=16)
        # Uniform logits -> uniform softmax -> weighted sum = reg_max/2 = 8.0
        x = torch.zeros(1, 17)
        out = integral(x)
        torch.testing.assert_close(out, torch.tensor([[8.0]]), atol=1e-4, rtol=1e-4)

    def test_peaked_distribution(self) -> None:
        """A peaked distribution at bin 5 should give ~5.0."""
        integral = Integral(reg_max=16)
        x = torch.full((1, 17), -100.0)
        x[0, 5] = 100.0  # very peaked at bin 5
        out = integral(x)
        assert abs(out.item() - 5.0) < 0.01

    def test_gradient_flows(self) -> None:
        integral = Integral(reg_max=16)
        x = torch.randn(4, 4 * 17, requires_grad=True)
        out = integral(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_batch_4_sides(self) -> None:
        """4*(reg_max+1) input should produce 4 distance values."""
        integral = Integral(reg_max=16)
        x = torch.randn(8, 68)  # 8 samples, 4 * 17 = 68
        out = integral(x)
        assert out.shape == (8, 4)
        assert torch.isfinite(out).all()
        # All outputs should be in [0, 16] since they're weighted sums
        assert (out >= 0).all()
        assert (out <= 16).all()


# ------------------------------------------------------------------ #
# 2. DistributionFocalLoss
# ------------------------------------------------------------------ #

class TestDistributionFocalLoss:
    def test_perfect_integer_target(self) -> None:
        """When target is exactly an integer, loss should equal CE at that bin."""
        dfl = DistributionFocalLoss(reduction="none")
        pred = torch.randn(1, 17)
        target = torch.tensor([5.0])
        loss = dfl(pred, target)
        # weight_left=1.0, weight_right=0.0 -> pure CE at bin 5
        expected = torch.nn.functional.cross_entropy(pred, torch.tensor([5]), reduction="none")
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)

    def test_midpoint_target(self) -> None:
        """Target 5.5 should give equal weight to bins 5 and 6."""
        dfl = DistributionFocalLoss(reduction="none")
        pred = torch.randn(1, 17)
        target = torch.tensor([5.5])
        loss = dfl(pred, target)
        ce5 = torch.nn.functional.cross_entropy(pred, torch.tensor([5]), reduction="none")
        ce6 = torch.nn.functional.cross_entropy(pred, torch.tensor([6]), reduction="none")
        expected = 0.5 * ce5 + 0.5 * ce6
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self) -> None:
        dfl = DistributionFocalLoss(reduction="sum")
        pred = torch.randn(4, 17, requires_grad=True)
        target = torch.rand(4) * 15  # targets in [0, 15]
        loss = dfl(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_reduction_modes(self) -> None:
        pred = torch.randn(4, 17)
        target = torch.rand(4) * 15
        none_loss = DistributionFocalLoss(reduction="none")(pred, target)
        sum_loss = DistributionFocalLoss(reduction="sum")(pred, target)
        mean_loss = DistributionFocalLoss(reduction="mean")(pred, target)
        torch.testing.assert_close(none_loss.sum(), sum_loss, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(none_loss.mean(), mean_loss, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ #
# 3. Distance encoding/decoding
# ------------------------------------------------------------------ #

class TestDistanceConversions:
    def test_roundtrip(self) -> None:
        """bbox2distance -> distance2bbox should recover the original box."""
        points = torch.tensor([[320.0, 320.0]])
        bbox = torch.tensor([[300.0, 280.0, 80.0, 120.0]])  # cxcywh
        stride = torch.tensor([8.0])

        dist = _bbox2distance(points, bbox, stride, reg_max=16)
        recovered = _distance2bbox(points, dist, stride)

        torch.testing.assert_close(recovered, bbox, atol=0.5, rtol=1e-3)

    def test_clamping(self) -> None:
        """Distances should be clamped to [0, reg_max)."""
        points = torch.tensor([[100.0, 100.0]])
        # Box far from anchor — distances would exceed reg_max
        bbox = torch.tensor([[300.0, 300.0, 400.0, 400.0]])
        stride = torch.tensor([8.0])

        dist = _bbox2distance(points, bbox, stride, reg_max=16)
        assert (dist >= 0).all()
        assert (dist < 16).all()

    def test_batch(self) -> None:
        """Batch of conversions should work."""
        points = torch.rand(10, 2) * 640
        bbox = torch.rand(10, 4) * 200 + 20
        bbox[:, :2] = points  # center at anchor for clean test
        stride = torch.full((10,), 8.0)

        dist = _bbox2distance(points, bbox, stride, reg_max=16)
        assert dist.shape == (10, 4)
        assert (dist >= 0).all()


# ------------------------------------------------------------------ #
# 4. DINOXHead with DFL — inference shape
# ------------------------------------------------------------------ #

class TestDINOXHeadDFL:
    @pytest.fixture
    def head_dfl(self) -> DINOXHead:
        return DINOXHead(
            num_classes=80, width=0.50, use_dfl=True, reg_max=16,
        )

    @pytest.fixture
    def head_nodfl(self) -> DINOXHead:
        return DINOXHead(
            num_classes=80, width=0.50, use_dfl=False,
        )

    @pytest.fixture
    def features(self) -> list[torch.Tensor]:
        return [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]

    @pytest.fixture
    def labels(self) -> torch.Tensor:
        labs = torch.zeros(2, 50, 5)
        labs[0, 0] = torch.tensor([0, 300, 300, 50, 50])
        labs[0, 1] = torch.tensor([1, 100, 200, 40, 60])
        labs[0, 2] = torch.tensor([2, 400, 400, 80, 80])
        labs[1, 0] = torch.tensor([5, 320, 320, 60, 60])
        labs[1, 1] = torch.tensor([10, 150, 150, 30, 30])
        return labs

    @pytest.fixture
    def imgs(self) -> torch.Tensor:
        return torch.randn(2, 3, 640, 640)

    def test_inference_shape_matches(
        self, head_dfl: DINOXHead, head_nodfl: DINOXHead, features: list[torch.Tensor],
    ) -> None:
        """DFL head should produce same inference output shape as non-DFL."""
        head_dfl.eval()
        head_nodfl.eval()
        with torch.no_grad():
            out_dfl = head_dfl(features)
            out_nodfl = head_nodfl(features)
        # Both should be (2, 8400, 85) = batch, anchors, 4+1+80
        assert out_dfl.shape == out_nodfl.shape == (2, 8400, 85)

    def test_inference_values_finite(
        self, head_dfl: DINOXHead, features: list[torch.Tensor],
    ) -> None:
        head_dfl.eval()
        with torch.no_grad():
            out = head_dfl(features)
        assert torch.isfinite(out).all()

    def test_training_loss_finite(
        self, head_dfl: DINOXHead, features: list[torch.Tensor],
        labels: torch.Tensor, imgs: torch.Tensor,
    ) -> None:
        head_dfl.train()
        result = head_dfl(features, labels, imgs)
        loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg = result
        for name, val in [("loss", loss), ("iou_loss", iou_loss),
                          ("obj_loss", obj_loss), ("cls_loss", cls_loss)]:
            t = val if torch.is_tensor(val) else torch.tensor(val)
            assert torch.isfinite(t).all(), f"{name} is not finite"

    def test_training_backward(
        self, head_dfl: DINOXHead, features: list[torch.Tensor],
        labels: torch.Tensor, imgs: torch.Tensor,
    ) -> None:
        head_dfl.train()
        loss, *_ = head_dfl(features, labels, imgs)
        assert loss.requires_grad
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head_dfl.parameters() if p.requires_grad
        )
        assert has_grad

    def test_dfl_head_more_params(
        self, head_dfl: DINOXHead, head_nodfl: DINOXHead,
    ) -> None:
        """DFL head should have more parameters (68 output channels vs 4)."""
        dfl_params = sum(p.numel() for p in head_dfl.parameters())
        nodfl_params = sum(p.numel() for p in head_nodfl.parameters())
        assert dfl_params > nodfl_params

    def test_zero_gt(
        self, head_dfl: DINOXHead, features: list[torch.Tensor], imgs: torch.Tensor,
    ) -> None:
        head_dfl.train()
        empty = torch.zeros(2, 50, 5)
        result = head_dfl(features, empty, imgs)
        loss, *_ = result
        t = loss if torch.is_tensor(loss) else torch.tensor(loss)
        assert torch.isfinite(t).all()

    def test_exp_config_loads(self) -> None:
        from yolox.exp import get_exp

        exp = get_exp(exp_name="dinox_s_dfl")
        model = exp.get_model()
        assert isinstance(model.head, DINOXHead)
        assert model.head.use_dfl is True
        assert model.head.reg_max == 16

    def test_exp_model_forward(self) -> None:
        from yolox.exp import get_exp

        exp = get_exp(exp_name="dinox_s_dfl")
        model = exp.get_model()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 640, 640))
        assert out.shape == (1, 8400, 85)
