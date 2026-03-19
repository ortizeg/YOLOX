"""Tests for a single training step (forward/backward/optimizer) using YOLOX on CPU."""

import pytest
import torch
import torch.nn as nn

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
from yolox.utils import LRScheduler


def _build_yolox_s():
    """Build a YOLOX-s model (depth=0.33, width=0.50)."""
    depth = 0.33
    width = 0.50
    num_classes = 80
    in_channels = [256, 512, 1024]

    backbone = YOLOPAFPN(depth=depth, width=width, in_channels=in_channels)
    head = YOLOXHead(num_classes=num_classes, width=width, in_channels=in_channels)
    model = YOLOX(backbone, head)
    return model


def _build_optimizer(model):
    """Build SGD optimizer with 3 param groups, mimicking yolox_base.py."""
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optimizer = torch.optim.SGD(pg0, lr=0.01, momentum=0.9, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": pg2})
    return optimizer


class TestTrainingStep:
    """Test a single forward/backward pass on CPU."""

    @pytest.fixture()
    def model(self):
        model = _build_yolox_s()
        model.train()
        return model

    @pytest.fixture()
    def inputs(self):
        torch.manual_seed(0)
        imgs = torch.randn(1, 3, 320, 320)
        # targets: [batch_idx, cls, cx, cy, w, h] - YOLOX expects (batch, max_labels, 5)
        targets = torch.zeros(1, 50, 5)
        # Add one fake target
        targets[0, 0] = torch.tensor([0, 160, 160, 50, 50])
        return imgs, targets

    def test_forward_produces_loss_dict(self, model, inputs):
        """Single forward pass in training mode produces a loss dict."""
        imgs, targets = inputs
        model.head.use_l1 = False

        outputs = model(imgs, targets)

        assert isinstance(outputs, dict), "Training forward should return a loss dict"
        assert "total_loss" in outputs, "Loss dict should contain 'total_loss'"
        assert outputs["total_loss"].dim() == 0, "total_loss should be a scalar"
        assert not torch.isnan(outputs["total_loss"]), "Loss should not be NaN"

    def test_backward_populates_gradients(self, model, inputs):
        """Backward pass succeeds and gradients are populated."""
        imgs, targets = inputs
        model.head.use_l1 = False

        outputs = model(imgs, targets)
        loss = outputs["total_loss"]
        loss.backward()

        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "At least some parameters should have non-zero gradients"

    def test_optimizer_step(self, model, inputs):
        """Optimizer step completes without error."""
        imgs, targets = inputs
        model.head.use_l1 = False
        optimizer = _build_optimizer(model)

        outputs = model(imgs, targets)
        loss = outputs["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify optimizer has 3 param groups
        assert len(optimizer.param_groups) == 3, "Should have 3 parameter groups"

    def test_lr_scheduler_integration(self, model):
        """LR scheduler (yoloxwarmcos) can be created and update_lr works."""
        optimizer = _build_optimizer(model)

        scheduler = LRScheduler(
            "yoloxwarmcos",
            lr=0.01,
            iters_per_epoch=100,
            total_epochs=10,
            warmup_epochs=5,
            warmup_lr_start=0.0,
            no_aug_epochs=2,
            min_lr_ratio=0.05,
        )

        # Update LR at iteration 0 should not crash
        lr = scheduler.update_lr(0)
        assert isinstance(lr, float), "update_lr should return a float"
        assert lr >= 0, "Learning rate should be non-negative"

        # Apply to optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
