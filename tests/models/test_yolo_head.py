import pytest
import torch

from yolox.models.yolo_head import YOLOXHead


@pytest.fixture
def head():
    """YOLOXHead configured for yolox-s (width=0.50, 80 classes)."""
    model = YOLOXHead(num_classes=80, width=0.50, strides=[8, 16, 32],
                       in_channels=[256, 512, 1024], act="silu", depthwise=False)
    return model


@pytest.fixture
def features():
    """Fake PAFPN outputs for yolox-s at 640x640 input."""
    return [
        torch.randn(2, 128, 80, 80),
        torch.randn(2, 256, 40, 40),
        torch.randn(2, 512, 20, 20),
    ]


@pytest.fixture
def labels():
    """Fake labels with shape (B, max_labels, 5).  First few rows filled."""
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
def imgs():
    """Fake input images (B, 3, 640, 640)."""
    return torch.randn(2, 3, 640, 640)


# ------------------------------------------------------------------ #
# 1. Inference output shape
# ------------------------------------------------------------------ #
class TestInference:
    def test_output_shape(self, head, features):
        head.eval()
        with torch.no_grad():
            out = head(features)
        # total_anchors = 80*80 + 40*40 + 20*20 = 8400
        assert out.shape == (2, 8400, 85), f"Expected (2, 8400, 85), got {out.shape}"

    def test_output_values_finite(self, head, features):
        head.eval()
        with torch.no_grad():
            out = head(features)
        assert torch.isfinite(out).all(), "Inference output contains non-finite values"


# ------------------------------------------------------------------ #
# 2. Training returns tuple of 6 values, all finite
# ------------------------------------------------------------------ #
class TestTrainingBasic:
    def test_returns_six_values(self, head, features, labels, imgs):
        head.train()
        result = head(features, labels, imgs)
        assert isinstance(result, (tuple, dict)), "Training forward should return a tuple or dict"
        # unpack
        loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg = result
        for name, val in [("loss", loss), ("iou_loss", iou_loss),
                          ("obj_loss", obj_loss), ("cls_loss", cls_loss),
                          ("l1_loss", l1_loss), ("num_fg", num_fg)]:
            assert torch.is_tensor(val) or isinstance(val, (float, int)), \
                f"{name} is not a tensor or scalar"

    def test_all_finite(self, head, features, labels, imgs):
        head.train()
        result = head(features, labels, imgs)
        loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg = result
        for name, val in [("loss", loss), ("iou_loss", iou_loss),
                          ("obj_loss", obj_loss), ("cls_loss", cls_loss),
                          ("l1_loss", l1_loss)]:
            t = val if torch.is_tensor(val) else torch.tensor(val)
            assert torch.isfinite(t).all(), f"{name} is not finite"


# ------------------------------------------------------------------ #
# 3. Loss is a scalar tensor with grad_fn
# ------------------------------------------------------------------ #
class TestLossGrad:
    def test_loss_has_grad_fn(self, head, features, labels, imgs):
        head.train()
        loss, *_ = head(features, labels, imgs)
        assert loss.requires_grad, "Loss should require grad"
        assert loss.grad_fn is not None, "Loss should have a grad_fn"
        assert loss.dim() == 0, "Loss should be a scalar (0-dim tensor)"

    def test_backward_runs(self, head, features, labels, imgs):
        head.train()
        loss, *_ = head(features, labels, imgs)
        loss.backward()
        # Check at least one parameter got a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in head.parameters() if p.requires_grad)
        assert has_grad, "No parameter received a gradient after backward"


# ------------------------------------------------------------------ #
# 4. Zero ground-truth objects (labels all zeros)
# ------------------------------------------------------------------ #
class TestZeroGT:
    def test_no_crash_on_zero_gt(self, head, features, imgs):
        head.train()
        empty_labels = torch.zeros(2, 50, 5)
        result = head(features, empty_labels, imgs)
        loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg = result
        # obj_loss should still be valid (all anchors are negatives)
        obj_val = obj_loss if torch.is_tensor(obj_loss) else torch.tensor(obj_loss)
        assert torch.isfinite(obj_val).all(), "obj_loss should be finite even with zero GT"


# ------------------------------------------------------------------ #
# 5. SimOTA matching (get_assignments) basic test
# ------------------------------------------------------------------ #
class TestGetAssignments:
    def test_get_assignments_runs(self, head, features, imgs):
        """Verify get_assignments returns expected outputs without error."""
        head.train()
        # Run a forward pass first to populate grids, then call get_assignments
        # We need to prepare the internal data that get_assignments expects.
        # Easiest: just run forward and check it doesn't crash (covered above).
        # Here we do a more direct test by extracting decoded outputs.
        head.eval()
        with torch.no_grad():
            # Run inference to populate internal state (grids, etc.)
            _ = head(features)

        head.train()
        # Build single-image inputs for get_assignments
        num_gt = 3
        # gt_bboxes in cxcywh format (center_x, center_y, width, height)
        gt_bboxes = torch.tensor([
            [300, 300, 40, 40],
            [100, 200, 40, 60],
            [400, 400, 80, 80],
        ], dtype=torch.float32)
        gt_classes = torch.tensor([0, 1, 2], dtype=torch.float32)

        # Fake per-anchor predictions
        total = 8400
        bbox_preds = torch.randn(total, 4).abs() * 100 + 10
        # cls_preds and obj_preds need batch dim since they're indexed by batch_idx
        cls_preds = torch.randn(1, total, 80)
        obj_preds = torch.randn(1, total, 1)
        # expanded_strides shape: [1, n_anchors_all]
        expanded_strides = torch.cat([
            torch.full((6400,), 8),
            torch.full((1600,), 16),
            torch.full((400,), 32),
        ], dim=0).float().unsqueeze(0)

        # x_shifts, y_shifts: grid cell coordinates, shape [1, n_anchors_all]
        shifts_80 = torch.arange(80).float()
        x80 = shifts_80.repeat(80)
        y80 = shifts_80.repeat_interleave(80)

        shifts_40 = torch.arange(40).float()
        x40 = shifts_40.repeat(40)
        y40 = shifts_40.repeat_interleave(40)

        shifts_20 = torch.arange(20).float()
        x20 = shifts_20.repeat(20)
        y20 = shifts_20.repeat_interleave(20)

        x_shifts = torch.cat([x80, x40, x20], dim=0).unsqueeze(0)
        y_shifts = torch.cat([y80, y40, y20], dim=0).unsqueeze(0)

        try:
            result = head.get_assignments(
                0,  # batch_idx
                num_gt,
                gt_bboxes,
                gt_classes,
                bbox_preds,
                expanded_strides,
                x_shifts,
                y_shifts,
                cls_preds,
                obj_preds,
            )
            # Should return (gt_matched_classes, fg_mask, pred_ious_this_matching,
            #                matched_gt_inds, num_fg)
            assert len(result) == 5, "get_assignments should return 5 values"
        except Exception as e:
            pytest.fail(f"get_assignments raised an exception: {e}")


# ------------------------------------------------------------------ #
# 6. get_l1_target computation test
# ------------------------------------------------------------------ #
class TestGetL1Target:
    def test_l1_target_shape(self, head):
        l1_target = torch.zeros(1, 4)
        gt = torch.tensor([[300, 300, 50, 50]], dtype=torch.float32)  # cxcywh
        stride = torch.tensor([8.0])
        x_offset = torch.tensor([37.0])  # grid x
        y_offset = torch.tensor([37.0])  # grid y
        result = head.get_l1_target(l1_target, gt, stride, x_offset, y_offset)
        assert result.shape == (1, 4), f"Expected shape (1, 4), got {result.shape}"
        assert torch.isfinite(result).all(), "L1 target should be finite"

    def test_l1_target_multiple(self, head):
        n = 5
        l1_target = torch.zeros(n, 4)
        gt = torch.rand(n, 4) * 200 + 10
        gt[:, 2:] = gt[:, 2:].abs() + 1  # ensure positive w, h
        stride = torch.full((n,), 16.0)
        x_offset = torch.arange(n).float()
        y_offset = torch.arange(n).float()
        result = head.get_l1_target(l1_target, gt, stride, x_offset, y_offset)
        assert result.shape == (n, 4)


# ------------------------------------------------------------------ #
# 7. initialize_biases test
# ------------------------------------------------------------------ #
class TestInitializeBiases:
    def test_biases_set(self):
        head = YOLOXHead(num_classes=80, width=0.50)
        # initialize_biases is called in __init__; check obj branch bias
        # The objectness conv bias should be initialized to a prior
        for cls_pred in head.cls_preds:
            bias = cls_pred.bias
            assert bias is not None, "cls_pred should have a bias"
            # bias should have been set by initialize_biases (not all zeros)

        for obj_pred in head.obj_preds:
            bias = obj_pred.bias
            assert bias is not None, "obj_pred should have a bias"


# ------------------------------------------------------------------ #
# 8. Depthwise variant
# ------------------------------------------------------------------ #
class TestDepthwise:
    def test_depthwise_inference(self):
        head = YOLOXHead(num_classes=80, width=0.50, depthwise=True)
        head.eval()
        features = [
            torch.randn(1, 128, 80, 80),
            torch.randn(1, 256, 40, 40),
            torch.randn(1, 512, 20, 20),
        ]
        with torch.no_grad():
            out = head(features)
        assert out.shape == (1, 8400, 85), f"Expected (1, 8400, 85), got {out.shape}"

    def test_depthwise_training(self):
        head = YOLOXHead(num_classes=80, width=0.50, depthwise=True)
        head.train()
        features = [
            torch.randn(2, 128, 80, 80),
            torch.randn(2, 256, 40, 40),
            torch.randn(2, 512, 20, 20),
        ]
        labels = torch.zeros(2, 50, 5)
        labels[0, 0] = torch.tensor([0, 300, 300, 50, 50])
        imgs = torch.randn(2, 3, 640, 640)
        result = head(features, labels, imgs)
        loss, iou_loss, obj_loss, cls_loss, l1_loss, num_fg = result
        assert torch.isfinite(torch.tensor(float(loss))), "Depthwise loss should be finite"

    def test_depthwise_has_fewer_params(self):
        regular = YOLOXHead(num_classes=80, width=0.50, depthwise=False)
        dw = YOLOXHead(num_classes=80, width=0.50, depthwise=True)
        regular_params = sum(p.numel() for p in regular.parameters())
        dw_params = sum(p.numel() for p in dw.parameters())
        assert dw_params < regular_params, \
            f"Depthwise ({dw_params}) should have fewer params than regular ({regular_params})"
