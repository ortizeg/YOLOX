import pytest
import torch

from yolox.models.losses import IOUloss


# ------------------------------------------------------------------ #
# 1. Perfect overlap -> loss near 0 for iou type
# ------------------------------------------------------------------ #
class TestPerfectOverlap:
    def test_identical_boxes_iou(self):
        loss_fn = IOUloss(reduction="none", loss_type="iou")
        boxes = torch.tensor([[50.0, 50.0, 30.0, 30.0]])  # cxcywh
        loss = loss_fn(boxes, boxes)
        assert loss.item() == pytest.approx(0.0, abs=1e-5), \
            f"Perfect overlap should give ~0 loss, got {loss.item()}"

    def test_identical_boxes_batch(self):
        loss_fn = IOUloss(reduction="none", loss_type="iou")
        boxes = torch.tensor([
            [50.0, 50.0, 30.0, 30.0],
            [100.0, 100.0, 60.0, 60.0],
            [200.0, 200.0, 10.0, 10.0],
        ])
        loss = loss_fn(boxes, boxes)
        assert loss.shape == (3,)
        assert (loss < 1e-5).all(), "All identical boxes should have ~0 loss"


# ------------------------------------------------------------------ #
# 2. No overlap -> loss near 1
# ------------------------------------------------------------------ #
class TestNoOverlap:
    def test_far_apart_boxes(self):
        loss_fn = IOUloss(reduction="none", loss_type="iou")
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target = torch.tensor([[1000.0, 1000.0, 10.0, 10.0]])
        loss = loss_fn(pred, target)
        # iou = 0 => loss = 1 - 0^2 = 1
        assert loss.item() == pytest.approx(1.0, abs=1e-4), \
            f"No-overlap iou loss should be ~1.0, got {loss.item()}"


# ------------------------------------------------------------------ #
# 3. GIoU loss tests
# ------------------------------------------------------------------ #
class TestGIoULoss:
    def test_giou_perfect_overlap(self):
        loss_fn = IOUloss(reduction="none", loss_type="giou")
        boxes = torch.tensor([[50.0, 50.0, 30.0, 30.0]])
        loss = loss_fn(boxes, boxes)
        # giou = 1 for perfect overlap => loss = 1 - 1 = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5), \
            f"GIoU perfect overlap should give ~0, got {loss.item()}"

    def test_giou_no_overlap(self):
        loss_fn = IOUloss(reduction="none", loss_type="giou")
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        target = torch.tensor([[1000.0, 1000.0, 10.0, 10.0]])
        loss = loss_fn(pred, target)
        assert loss.item() > 0, "GIoU loss for non-overlapping boxes should be > 0"

    def test_giou_bounded(self):
        """GIoU is in [-1, 1], so loss = 1 - giou is in [0, 2]."""
        loss_fn = IOUloss(reduction="none", loss_type="giou")
        pred = torch.rand(100, 4) * 200 + 1
        pred[:, 2:] = pred[:, 2:].abs() + 1
        target = torch.rand(100, 4) * 200 + 1
        target[:, 2:] = target[:, 2:].abs() + 1
        loss = loss_fn(pred, target)
        assert (loss >= -0.01).all(), "GIoU loss should be >= 0 (approx)"
        assert (loss <= 2.01).all(), "GIoU loss should be <= 2"


# ------------------------------------------------------------------ #
# 4. Gradient validity
# ------------------------------------------------------------------ #
class TestGradient:
    def test_backward_iou(self):
        loss_fn = IOUloss(reduction="mean", loss_type="iou")
        pred = torch.tensor([[50.0, 50.0, 30.0, 30.0]], requires_grad=True)
        target = torch.tensor([[60.0, 60.0, 30.0, 30.0]])
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None, "Gradient should exist"
        assert torch.isfinite(pred.grad).all(), "Gradient should be finite"

    def test_backward_giou(self):
        loss_fn = IOUloss(reduction="mean", loss_type="giou")
        pred = torch.tensor([[50.0, 50.0, 30.0, 30.0]], requires_grad=True)
        target = torch.tensor([[60.0, 60.0, 30.0, 30.0]])
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None, "Gradient should exist"
        assert torch.isfinite(pred.grad).all(), "Gradient should be finite"


# ------------------------------------------------------------------ #
# 5. Reduction modes: "none", "mean", "sum"
# ------------------------------------------------------------------ #
class TestReduction:
    @pytest.fixture
    def boxes(self):
        pred = torch.tensor([
            [50.0, 50.0, 30.0, 30.0],
            [100.0, 100.0, 40.0, 40.0],
            [200.0, 200.0, 20.0, 20.0],
        ])
        target = torch.tensor([
            [55.0, 55.0, 30.0, 30.0],
            [110.0, 110.0, 40.0, 40.0],
            [210.0, 210.0, 20.0, 20.0],
        ])
        return pred, target

    def test_none_reduction(self, boxes):
        pred, target = boxes
        loss_fn = IOUloss(reduction="none", loss_type="iou")
        loss = loss_fn(pred, target)
        assert loss.shape == (3,), f"Expected shape (3,), got {loss.shape}"

    def test_mean_reduction(self, boxes):
        pred, target = boxes
        loss_none = IOUloss(reduction="none", loss_type="iou")(pred, target)
        loss_mean = IOUloss(reduction="mean", loss_type="iou")(pred, target)
        assert loss_mean.dim() == 0, "Mean reduction should produce a scalar"
        assert loss_mean.item() == pytest.approx(loss_none.mean().item(), abs=1e-5)

    def test_sum_reduction(self, boxes):
        pred, target = boxes
        loss_none = IOUloss(reduction="none", loss_type="iou")(pred, target)
        loss_sum = IOUloss(reduction="sum", loss_type="iou")(pred, target)
        assert loss_sum.dim() == 0, "Sum reduction should produce a scalar"
        assert loss_sum.item() == pytest.approx(loss_none.sum().item(), abs=1e-5)


# ------------------------------------------------------------------ #
# 6. Batch of boxes
# ------------------------------------------------------------------ #
class TestBatch:
    def test_large_batch(self):
        loss_fn = IOUloss(reduction="none", loss_type="iou")
        n = 256
        pred = torch.rand(n, 4) * 200 + 10
        pred[:, 2:] = pred[:, 2:].abs() + 1
        target = torch.rand(n, 4) * 200 + 10
        target[:, 2:] = target[:, 2:].abs() + 1
        loss = loss_fn(pred, target)
        assert loss.shape == (n,), f"Expected shape ({n},), got {loss.shape}"
        assert torch.isfinite(loss).all(), "All losses should be finite"

    def test_large_batch_giou(self):
        loss_fn = IOUloss(reduction="mean", loss_type="giou")
        n = 256
        pred = torch.rand(n, 4) * 200 + 10
        pred[:, 2:] = pred[:, 2:].abs() + 1
        target = torch.rand(n, 4) * 200 + 10
        target[:, 2:] = target[:, 2:].abs() + 1
        loss = loss_fn(pred, target)
        assert loss.dim() == 0, "Should be scalar with mean reduction"
        assert torch.isfinite(loss), "Mean loss should be finite"
