import os

import pytest
import torch
import torch.nn as nn

from yolox.utils.checkpoint import load_ckpt, save_checkpoint


def _make_model():
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))


class TestLoadCkpt:
    def test_round_trip(self, tmp_path):
        model = _make_model()
        state = {"model": model.state_dict()}
        path = tmp_path / "ckpt.pth"
        torch.save(state, path)

        loaded_state = torch.load(path, map_location="cpu")
        new_model = _make_model()
        # Scramble weights to ensure load actually changes them
        with torch.no_grad():
            for p in new_model.parameters():
                p.fill_(0.0)

        # load_ckpt expects the state_dict directly, not wrapped in {"model": ...}
        load_ckpt(new_model, loaded_state["model"])

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch on {n1}"

    def test_mismatched_shapes_warns_and_skips(self):
        model = _make_model()
        ckpt = model.state_dict()

        # Create model with different shape
        different_model = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2))
        # Should not raise, just warn and skip mismatched keys
        loaded = load_ckpt(different_model, ckpt)
        assert loaded is not None

    def test_missing_keys_warns_and_skips(self):
        model = _make_model()
        # Empty checkpoint
        ckpt = {}
        loaded = load_ckpt(model, ckpt)
        assert loaded is not None


class TestSaveCheckpoint:
    def test_creates_directory(self, tmp_path):
        model = _make_model()
        save_dir = str(tmp_path / "new_dir")
        state = {"model": model.state_dict()}
        save_checkpoint(state, is_best=False, save_dir=save_dir, model_name="test")
        assert os.path.isdir(save_dir)
        assert os.path.exists(os.path.join(save_dir, "test_ckpt.pth"))

    def test_is_best_creates_best_ckpt(self, tmp_path):
        model = _make_model()
        save_dir = str(tmp_path)
        state = {"model": model.state_dict()}
        save_checkpoint(state, is_best=True, save_dir=save_dir, model_name="test")
        assert os.path.exists(os.path.join(save_dir, "best_ckpt.pth"))

    def test_not_best_does_not_create_best_ckpt(self, tmp_path):
        model = _make_model()
        save_dir = str(tmp_path)
        state = {"model": model.state_dict()}
        save_checkpoint(state, is_best=False, save_dir=save_dir, model_name="test")
        assert not os.path.exists(os.path.join(save_dir, "best_ckpt.pth"))
