"""Tier A: lean_ckpt changes which checkpoint files are written and what they hold, never the training maths.

The trainer is simulated (a tiny model + AdamW stepped once per "epoch", a scripted eval AP); the real
``Trainer.after_epoch`` / ``evaluate_and_save_model`` / ``save_ckpt*`` / ``resume_train`` methods are bound to it.
"""

import os
import types

import pytest
import torch
from torch import nn

from yolox.core.trainer import Trainer
from yolox.utils import checkpoint as ckpt_mod
from yolox.utils import load_ckpt

from .conftest import flags

_METHODS = ["after_epoch", "evaluate_and_save_model", "save_ckpt", "save_ckpt_lean", "_latest_due",
            "_join_ckpt_writer", "resume_train"]


def tiny_model():
    return nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Conv2d(8, 4, 1))


def make_trainer(tmp, max_epoch=12, eval_interval=5, no_aug_epochs=3, save_interval=None, history=True,
                 aps=None):
    torch.manual_seed(0)
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    aps = list(aps or [])

    def fake_eval(evalmodel, evaluator, is_distributed, return_outputs=False):
        return (aps.pop(0) if aps else 0.0, 0.0, "summary"), None

    exp = types.SimpleNamespace(eval_interval=eval_interval, no_aug_epochs=no_aug_epochs, eval=fake_eval)
    if save_interval is not None:
        exp.save_interval = save_interval
    t = types.SimpleNamespace(
        rank=0, use_model_ema=False, model=model, optimizer=opt, best_ap=0.0, epoch=0, max_epoch=max_epoch,
        file_name=str(tmp), exp=exp, evaluator=None, is_distributed=False, save_history_ckpt=history,
        device="cpu", args=types.SimpleNamespace(logger="none", resume=False, ckpt=None, start_epoch=None))
    for m in _METHODS:
        setattr(t, m, types.MethodType(getattr(Trainer, m), t))
    return t


def train_step(t):
    t.model.train()
    x = torch.rand(2, 3, 16, 16)
    t.optimizer.zero_grad()
    t.model(x).square().mean().backward()
    t.optimizer.step()


def run(t, epochs=None, on_epoch=None):
    """Simulated train_in_epoch: one optimizer step per epoch, the no-aug switch to eval_interval 1 as in
    before_epoch, then the real after_epoch."""
    for t.epoch in range(epochs if epochs is not None else t.max_epoch):
        if t.epoch + 1 == t.max_epoch - t.exp.no_aug_epochs:
            t.exp.eval_interval = 1
        train_step(t)
        t.after_epoch()
        t._join_ckpt_writer()
        if on_epoch:
            on_epoch(t)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def flat(x, prefix=""):
    if torch.is_tensor(x):
        return {prefix: x}
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flat(v, f"{prefix}/{k}"))
        return out
    if isinstance(x, (list, tuple)):
        out = {}
        for i, v in enumerate(x):
            out.update(flat(v, f"{prefix}/{i}"))
        return out
    return {prefix: x}


def same(a, b):
    fa, fb = flat(a), flat(b)
    if fa.keys() != fb.keys():
        return False
    return all(torch.equal(fa[k], fb[k]) if torch.is_tensor(fa[k]) else fa[k] == fb[k] for k in fa)


def ckpt_files(d):
    return sorted(f for f in os.listdir(d) if f.endswith("_ckpt.pth"))


ALL_FILES = sorted(["latest_ckpt.pth", "last_epoch_ckpt.pth", "best_ckpt.pth"]
                   + [f"epoch_{e}_ckpt.pth" for e in (5, 9, 10, 11, 12)])
APS = [0.1, 0.3, 0.2, 0.4, 0.4, 0.35]  # evals at epochs 5, 9, 10, 11, 12 (max 12, eval 5, no-aug 3)


def test_flag_off_original_files(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("lean writer used with the flag off")

    monkeypatch.setattr("yolox.core.trainer.save_checkpoint_lean", boom)
    with flags(""):
        t = make_trainer(tmp_path, save_interval=4, aps=APS)
        latest_epochs = []
        run(t, on_epoch=lambda t: latest_epochs.append(load(tmp_path / "latest_ckpt.pth")["start_epoch"]))
    assert latest_epochs == list(range(1, 13))  # every epoch, save_interval ignored
    assert ckpt_files(tmp_path) == ALL_FILES
    for f in ckpt_files(tmp_path):
        c = load(tmp_path / f)
        assert set(c) == {"start_epoch", "model", "optimizer", "best_ap", "curr_ap"}, f
    assert load(tmp_path / "best_ckpt.pth")["start_epoch"] == 11  # last strict improvement (0.4 at epoch 11)
    assert not os.path.samefile(tmp_path / "latest_ckpt.pth", tmp_path / "last_epoch_ckpt.pth")


def test_save_interval_schedule(tmp_path):
    with flags("lean_ckpt"):
        t = make_trainer(tmp_path, save_interval=4, aps=APS)
        written = []

        def rec(t):
            p = tmp_path / "latest_ckpt.pth"
            if p.exists() and load(p)["start_epoch"] == t.epoch + 1:
                written.append(t.epoch + 1)

        run(t, on_epoch=rec)
    # 4, 8 (save_interval), 5 (eval), 9 (no-aug boundary + eval), 10-12 (eval), 12 (final)
    assert written == [4, 5, 8, 9, 10, 11, 12]


def test_save_interval_default_every_epoch(tmp_path):
    with flags("lean_ckpt"):
        t = make_trainer(tmp_path, aps=APS)  # exp without save_interval: getattr default 1
        written = []
        run(t, on_epoch=lambda t: written.append(load(tmp_path / "latest_ckpt.pth")["start_epoch"]))
    assert written == list(range(1, 13))


def test_final_epoch_written_without_eval(tmp_path):
    with flags("lean_ckpt"):
        t = make_trainer(tmp_path, max_epoch=7, eval_interval=100, no_aug_epochs=0, save_interval=3)
        run(t)
    assert load(tmp_path / "latest_ckpt.pth")["start_epoch"] == 7


def test_lean_files_resume_and_eval_load(tmp_path):
    snaps = {}
    with flags("lean_ckpt"):
        t = make_trainer(tmp_path, save_interval=4, aps=APS)
        run(t, on_epoch=lambda t: snaps.__setitem__(
            t.epoch + 1, {k: v.clone() for k, v in t.model.state_dict().items()}))
    assert ckpt_files(tmp_path) == ALL_FILES

    # resume from latest restores model, optimizer and start_epoch (the original resume_train path)
    latest = load(tmp_path / "latest_ckpt.pth")
    assert set(latest) == {"start_epoch", "model", "optimizer", "best_ap", "curr_ap"}
    r = make_trainer(tmp_path / "unused")
    r.args = types.SimpleNamespace(logger="none", resume=True, ckpt=str(tmp_path / "latest_ckpt.pth"),
                                   start_epoch=None)
    model = r.resume_train(r.model)
    assert r.start_epoch == 12
    assert r.best_ap == pytest.approx(0.4)
    assert same(model.state_dict(), t.model.state_dict())
    assert same(r.optimizer.state_dict(), t.optimizer.state_dict())

    # last_epoch: the same file as latest at that moment (hardlink)
    assert os.path.samefile(tmp_path / "latest_ckpt.pth", tmp_path / "last_epoch_ckpt.pth")

    # best / epoch_N: weights-only, load through load_ckpt and the tools/eval.py path, same weights as that epoch
    for name, epoch in [("best", 11)] + [(f"epoch_{e}", e) for e in (5, 9, 10, 11, 12)]:
        path = tmp_path / f"{name}_ckpt.pth"
        c = load(path)
        assert set(c) == {"start_epoch", "model", "best_ap", "curr_ap"}, name
        assert c["start_epoch"] == epoch
        m1, m2 = tiny_model(), tiny_model()
        load_ckpt(m1, c["model"])
        m2.load_state_dict(c["model"])  # tools/eval.py, demo.py, trt.py
        assert same(m1.state_dict(), snaps[epoch]) and same(m2.state_dict(), snaps[epoch]), name
        assert path.stat().st_size < (tmp_path / "latest_ckpt.pth").stat().st_size
    assert os.path.samefile(tmp_path / "best_ckpt.pth", tmp_path / "epoch_11_ckpt.pth")


def test_last_epoch_survives_next_latest(tmp_path):
    """latest is replaced atomically (new inode), so the hardlinked last_epoch keeps the eval epoch's content."""
    with flags("lean_ckpt"):
        t = make_trainer(tmp_path, max_epoch=20, eval_interval=5, no_aug_epochs=0, save_interval=1, aps=[0.1, 0.2])
        run(t, epochs=7)
    assert load(tmp_path / "latest_ckpt.pth")["start_epoch"] == 7
    assert load(tmp_path / "last_epoch_ckpt.pth")["start_epoch"] == 5
    assert not os.path.samefile(tmp_path / "latest_ckpt.pth", tmp_path / "last_epoch_ckpt.pth")
    assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))


def test_link_fallback_copy(tmp_path, monkeypatch):
    def no_link(src, dst):
        raise OSError("links not supported")

    monkeypatch.setattr(ckpt_mod.os, "link", no_link)
    with flags("lean_ckpt"):
        t = make_trainer(tmp_path, aps=APS)
        run(t)
    assert not os.path.samefile(tmp_path / "latest_ckpt.pth", tmp_path / "last_epoch_ckpt.pth")
    assert same(load(tmp_path / "latest_ckpt.pth"), load(tmp_path / "last_epoch_ckpt.pth"))
    assert same(load(tmp_path / "best_ckpt.pth"), load(tmp_path / "epoch_11_ckpt.pth"))


def test_composes_with_async_ckpt(tmp_path):
    a, b = tmp_path / "sync", tmp_path / "async"
    with flags("lean_ckpt"):
        run(make_trainer(a, save_interval=4, aps=APS))
    with flags("lean_ckpt,async_ckpt"):
        t = make_trainer(b, save_interval=4, aps=APS)
        for t.epoch in range(t.max_epoch):
            if t.epoch + 1 == t.max_epoch - t.exp.no_aug_epochs:
                t.exp.eval_interval = 1
            train_step(t)
            t.after_epoch()
            keep = [p.detach().clone() for p in t.model.parameters()]
            with torch.no_grad():  # training goes on while the writer runs: the snapshot must not change
                for p in t.model.parameters():
                    p.add_(1.0)
            t._join_ckpt_writer()
            with torch.no_grad():  # exact restore (p + 1 - 1 is not p in float)
                for p, k in zip(t.model.parameters(), keep):
                    p.copy_(k)
    assert ckpt_files(a) == ckpt_files(b)
    for f in ckpt_files(a):
        assert same(load(a / f), load(b / f)), f
    assert os.path.samefile(b / "latest_ckpt.pth", b / "last_epoch_ckpt.pth")
    assert not any(f.endswith(".tmp") for f in os.listdir(b))
