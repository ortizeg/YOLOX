"""Extended tests for yolox/utils/model_utils.py."""

import pytest
import torch
import torch.nn as nn

from yolox.utils.model_utils import (
    adjust_status,
    freeze_module,
    fuse_conv_and_bn,
    fuse_model,
)


class TestFuseConvAndBn:
    """Tests for fuse_conv_and_bn."""

    def test_fused_conv_matches_original_output(self):
        """Output of fused conv matches original conv+bn output within tolerance."""
        torch.manual_seed(42)
        conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        bn = nn.BatchNorm2d(16)

        # Run a few batches through to populate running stats
        bn.train()
        for _ in range(5):
            x = torch.randn(4, 3, 8, 8)
            bn(conv(x))

        bn.eval()
        conv.eval()

        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            original_output = bn(conv(x))
            fused = fuse_conv_and_bn(conv, bn)
            fused_output = fused(x)

        torch.testing.assert_close(fused_output, original_output, atol=1e-5, rtol=1e-5)


class TestFuseModel:
    """Tests for fuse_model."""

    def test_fused_model_produces_same_output(self):
        """Model still produces the same output after fusion (eval mode)."""
        from yolox.models.network_blocks import BaseConv

        torch.manual_seed(42)
        model = nn.Sequential(
            BaseConv(3, 16, ksize=3, stride=1),
            BaseConv(16, 32, ksize=3, stride=1),
        )
        model.eval()

        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            original_output = model(x)

        fuse_model(model)

        with torch.no_grad():
            fused_output = model(x)

        torch.testing.assert_close(fused_output, original_output, atol=1e-4, rtol=1e-4)


class TestFreezeModule:
    """Tests for freeze_module."""

    def test_all_params_frozen(self):
        """All params have requires_grad=False after freezing."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        freeze_module(model)

        for param in model.parameters():
            assert not param.requires_grad, "All parameters should be frozen"

    def test_freeze_with_name_filter(self):
        """Only matching params are frozen when name filter is provided."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        # freeze_module with name freezes only params whose name contains the string
        freeze_module(model, name="0")

        # Params in module "0" should be frozen
        for name, param in model.named_parameters():
            if "0" in name:
                assert not param.requires_grad, f"Param {name} should be frozen"
            else:
                assert param.requires_grad, f"Param {name} should NOT be frozen"

    def test_module_set_to_eval(self):
        """Frozen module is set to eval mode."""
        model = nn.Sequential(nn.Linear(10, 5), nn.BatchNorm1d(5))
        model.train()
        freeze_module(model)
        assert not model.training, "Module should be in eval mode after freeze"


class TestAdjustStatus:
    """Tests for adjust_status context manager."""

    def test_eval_inside_context(self):
        """Module is eval inside context when training=False."""
        model = nn.Linear(10, 5)
        model.train()
        assert model.training

        with adjust_status(model, training=False):
            assert not model.training, "Module should be in eval mode inside context"

        assert model.training, "Module should be restored to train mode after context"

    def test_train_inside_context(self):
        """Module is train inside context when training=True."""
        model = nn.Linear(10, 5)
        model.eval()
        assert not model.training

        with adjust_status(model, training=True):
            assert model.training, "Module should be in train mode inside context"

        assert not model.training, "Module should be restored to eval mode after context"
