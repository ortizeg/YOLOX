import copy

import torch
import torch.nn as nn

from yolox.utils.ema import ModelEMA, is_parallel


def _make_model():
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))


class TestModelEMA:
    def test_initialization_eval_mode_and_frozen_params(self):
        model = _make_model()
        ema = ModelEMA(model, decay=0.9999)
        assert not ema.ema.training
        for p in ema.ema.parameters():
            assert not p.requires_grad

    def test_initial_params_match_model(self):
        model = _make_model()
        ema = ModelEMA(model, decay=0.9999)
        for p_model, p_ema in zip(model.parameters(), ema.ema.parameters()):
            assert torch.allclose(p_model, p_ema)

    def test_update_shifts_toward_model(self):
        model = _make_model()
        ema = ModelEMA(model, decay=0.9999)

        # Record original EMA params
        original_ema_params = [p.clone() for p in ema.ema.parameters()]

        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 10.0)

        ema.update(model)

        # EMA params should have shifted (not equal to originals anymore)
        for orig, current in zip(original_ema_params, ema.ema.parameters()):
            # After one update, EMA should have moved slightly toward model
            assert not torch.allclose(orig, current, atol=1e-6)

    def test_multiple_updates_converge(self):
        model = _make_model()
        ema = ModelEMA(model, decay=0.9999)

        # Change model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)

        # Many updates should bring EMA close to model
        for _ in range(10000):
            ema.update(model)

        for p_model, p_ema in zip(model.parameters(), ema.ema.parameters()):
            assert torch.allclose(p_model, p_ema, atol=1e-2)


class TestIsParallel:
    def test_regular_model_returns_false(self):
        model = _make_model()
        assert is_parallel(model) is False
