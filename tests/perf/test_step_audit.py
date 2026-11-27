"""Step-audit hooks (yolox/utils/step_audit.py): off by default, a stage profile summary that is right on a known
event list, YOLOX_AUDIT_ITERS stopping a real (tiny, CPU) training loop, and the sync recorder's capture/dedupe.

The trainer is a real ``Trainer`` built with ``__new__`` (``before_train`` needs CUDA) around a tiny YOLOX on CPU, so
``train_in_epoch`` / ``train_in_iter`` / ``train_one_iter`` / ``after_iter`` / ``after_train`` are the real code.
"""

import json
import types
import warnings

import pytest
import torch

from yolox.core.trainer import Trainer
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import MeterBuffer, step_audit

ENV = ("YOLOX_AUDIT_ITERS", "YOLOX_SYNC_AUDIT", "YOLOX_PROFILE", "YOLOX_MEM_CAP_GB", "YOLOX_AUDIT_OUT",
       "YOLOX_SYNC_AUDIT_WARMUP", "YOLOX_SYNC_AUDIT_STEPS", "YOLOX_AUDIT_SPAN_EPOCHS")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    yield
    step_audit._MODE = 0
    step_audit._STACK.clear()


class _Prefetcher:
    def __init__(self, n=4):
        g = torch.Generator().manual_seed(0)
        self.batches = []
        for _ in range(n):
            inps = torch.rand(2, 3, 64, 64, generator=g) * 255
            t = torch.zeros(2, 6, 5)
            t[:, :3, 0] = torch.randint(0, 2, (2, 3), generator=g).float()
            t[:, :3, 1:3] = torch.rand(2, 3, 2, generator=g) * 40 + 12
            t[:, :3, 3:5] = torch.rand(2, 3, 2, generator=g) * 20 + 8
            self.batches.append((inps, t))
        self.i = 0

    def next(self):
        b = self.batches[self.i % len(self.batches)]
        self.i += 1
        return b[0].clone(), b[1].clone()


def make_trainer(tmp_path, max_iter=6, max_epoch=2):
    torch.manual_seed(0)
    model = YOLOX(YOLOPAFPN(depth=0.33, width=0.125), YOLOXHead(2, width=0.125))
    model.train()
    exp = types.SimpleNamespace(
        preprocess=lambda i, t, s: (i, t), print_interval=1000, eval_interval=1, no_aug_epochs=0,
        random_resize=lambda loader, epoch, rank, dist: (64, 64))
    t = Trainer.__new__(Trainer)
    t.__dict__.update(
        exp=exp, args=types.SimpleNamespace(logger="tensorboard", ckpt=None, exp_file=None), max_epoch=max_epoch,
        amp_training=False, amp_dtype=torch.float16, scaler=torch.amp.GradScaler("cuda", enabled=False),
        is_distributed=False, rank=0, local_rank=0, device="cpu", use_model_ema=False, save_history_ckpt=False,
        data_type=torch.float32, input_size=(64, 64), best_ap=0, meter=MeterBuffer(window_size=1000),
        file_name=str(tmp_path / "run"), model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        lr_scheduler=types.SimpleNamespace(update_lr=lambda it: 0.01), prefetcher=_Prefetcher(),
        channels_last=False, start_epoch=0, max_iter=max_iter, no_aug=False, train_loader=None,
        audit=step_audit.from_env(0, str(tmp_path / "run")), _audit_stop=False)
    t.calls = types.SimpleNamespace(iters=0, epochs_ended=0, before_epoch=0)

    real_one_iter = t.train_one_iter

    def one_iter():
        t.calls.iters += 1
        real_one_iter()

    def before_epoch():
        t.calls.before_epoch += 1

    def after_epoch():
        t.calls.epochs_ended += 1

    t.train_one_iter, t.before_epoch, t.after_epoch = one_iter, before_epoch, after_epoch
    return t


def weights(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


# --------------------------------------------------------------------------------------------- off by default
def test_hooks_off_when_env_unset(tmp_path):
    assert step_audit.from_env(0, str(tmp_path)) is None
    assert step_audit.stage("forward") is step_audit._NULL  # shared no-op context
    with step_audit.stage("forward"):
        assert step_audit.current_stage() is None
    t = make_trainer(tmp_path, max_iter=3, max_epoch=2)
    assert t.audit is None
    t.train_in_epoch()
    t.after_train()
    assert (t.calls.iters, t.calls.epochs_ended) == (6, 2)  # every iteration and epoch end, as without the hooks
    assert step_audit._MODE == 0
    assert not (tmp_path / "run").exists()  # no audit files


def test_stage_tracking_does_not_change_training(tmp_path, monkeypatch):
    ref = make_trainer(tmp_path / "a", max_iter=3, max_epoch=1)
    ref.train_in_epoch()
    monkeypatch.setenv("YOLOX_AUDIT_OUT", str(tmp_path / "audit"))
    monkeypatch.setenv("YOLOX_AUDIT_ITERS", "100")
    t = make_trainer(tmp_path / "b", max_iter=3, max_epoch=1)
    step_audit._MODE = 2  # ranges on (record_function), as with YOLOX_PROFILE
    seen = []
    real = step_audit._Stage.__enter__

    def spy(self):
        seen.append(self.name)
        return real(self)

    monkeypatch.setattr(step_audit._Stage, "__enter__", spy)
    t.train_in_epoch()
    assert torch.equal(weights(ref.model), weights(t.model))
    for s in ("data_fetch", "preprocess", "forward", "backbone_neck", "head", "head_targets", "head_loss",
              "backward", "optimizer_step", "lr_update", "meters", "log"):
        assert s in seen, s


# --------------------------------------------------------------------------------------------- YOLOX_AUDIT_ITERS
def test_audit_iters_stops_training_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLOX_AUDIT_ITERS", "5")
    monkeypatch.setenv("YOLOX_AUDIT_OUT", str(tmp_path / "audit"))
    t = make_trainer(tmp_path, max_iter=100, max_epoch=3)
    w0 = weights(t.model).clone()
    t.train_in_epoch()
    assert t.calls.iters == 5 and t.audit.count == 5
    assert t.calls.epochs_ended == 0 and t.calls.before_epoch == 1  # no after_epoch: no checkpoint, no eval
    assert not torch.equal(w0, weights(t.model))  # it really trained
    t.save_ckpt("latest")
    t.save_ckpt_lean(full=["latest"])
    assert not list((tmp_path / "run").glob("*.pth"))
    t.after_train()
    rec = json.loads((tmp_path / "audit" / "mem_rank0.json").read_text())
    assert rec["iters_done"] == 5 and rec["oom"] is False


def test_audit_iters_longer_than_epoch_stops_at_epoch_end(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLOX_AUDIT_ITERS", "50")
    monkeypatch.setenv("YOLOX_AUDIT_OUT", str(tmp_path / "audit"))
    t = make_trainer(tmp_path, max_iter=4, max_epoch=3)
    t.train_in_epoch()
    assert (t.calls.iters, t.calls.epochs_ended) == (4, 0)


def test_audit_span_epochs_counts_across_epochs(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLOX_AUDIT_ITERS", "10")
    monkeypatch.setenv("YOLOX_AUDIT_SPAN_EPOCHS", "1")
    monkeypatch.setenv("YOLOX_AUDIT_OUT", str(tmp_path / "audit"))
    t = make_trainer(tmp_path, max_iter=4, max_epoch=5)
    t.train_in_epoch()
    # 4 + 4 + 2 iterations over three epochs, and never an end-of-epoch checkpoint or evaluation
    assert (t.calls.iters, t.audit.count, t.calls.before_epoch, t.calls.epochs_ended) == (10, 10, 3, 0)


def test_oom_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLOX_MEM_CAP_GB", "24")
    monkeypatch.setenv("YOLOX_AUDIT_OUT", str(tmp_path / "audit"))
    a = step_audit.from_env(3, str(tmp_path))
    a.finish(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    rec = json.loads((tmp_path / "audit" / "mem_rank3.json").read_text())
    assert rec["oom"] is True and rec["rank"] == 3 and rec["mem_cap_gb"] == 24.0


# --------------------------------------------------------------------------------------------- profile summary
def _x(name, ts, dur, cat="user_annotation", corr=None):
    e = {"ph": "X", "name": name, "cat": cat, "ts": ts, "dur": dur, "pid": 1, "tid": 1}
    if corr is not None:
        e["args"] = {"correlation": corr}
    return e


def synthetic_events():
    ev = [
        _x("ProfilerStep#1", 0, 10000), _x("ProfilerStep#2", 10000, 10000),
        _x("audit/forward", 1000, 4000), _x("audit/forward", 11000, 4000),
        _x("audit/forward", 1000, 9000, cat="gpu_user_annotation"),  # GPU-side copy: ignored
        _x("audit/head", 1500, 2000), _x("audit/head_targets", 1600, 500), _x("audit/head_loss", 2500, 500),
        _x("audit/backward", 5000, 3000), _x("audit/backward", 15000, 3000),
        _x("audit/forward", 30000, 1000),  # outside the profiled window: ignored
        _x("cudaStreamSynchronize", 8500, 500, cat="cuda_runtime"),
    ]
    # (corr, launch ts, kernel ts, kernel dur, name)
    for corr, lt, kt, kd, name in [(1, 1500, 2000, 2000, "gemm"), (2, 6000, 7000, 3000, "conv_bwd"),
                                   (3, 11500, 12000, 2000, "gemm"), (4, 16000, 16500, 1000, "conv_bwd"),
                                   (5, 16100, 17000, 1000, "ncclKernel_AllReduce"),
                                   (7, 24000, 25000, 1000, "late")]:
        ev += [_x("cudaLaunchKernel", lt, 10, cat="cuda_runtime", corr=corr), _x(name, kt, kd, cat="kernel", corr=corr)]
    ev += [_x("cudaMemcpyAsync", 500, 10, cat="cuda_runtime", corr=6), _x("Memcpy HtoD", 0, 1000, cat="gpu_memcpy",
                                                                           corr=6)]
    return ev


def test_summary_busy_fraction_and_stages():
    s = step_audit.summarize_trace(synthetic_events())
    approx = lambda v: pytest.approx(v, abs=1e-9)
    assert s["steps"] == 2
    assert s["step_wall_ms"] == approx(10.0) and s["window_ms"] == approx(20.0)
    # kernel union in [0, 20000]: [2,4] + [7,10] + [12,14] + [16.5,18] ms = 8.5 ms
    assert s["gpu_busy_fraction"] == approx(8.5 / 20)
    assert s["gpu_busy_ms_per_step"] == approx(4.25)
    assert s["gpu_busy_fraction_incl_memcpy"] == approx(9.5 / 20)
    assert s["gpu_busy_fraction_excl_nccl"] == approx(8.0 / 20)
    assert s["per_step_busy_fraction"] == [approx(0.5), approx(0.35)]
    assert s["kernels_per_step"] == approx(2.5)
    assert s["mean_kernel_us"] == approx(9000 / 5)
    assert s["nccl_kernel_ms_per_step"] == approx(0.5)
    assert s["host_sync"]["cudaStreamSynchronize"] == {"calls_per_step": approx(0.5), "ms_per_step": approx(0.25)}
    st = s["stages"]
    assert st["forward"]["cpu_ms"] == approx(4.0) and st["forward"]["cuda_ms"] == approx(2.0)
    assert st["forward"]["calls_per_step"] == approx(1.0)
    assert st["backward"]["cpu_ms"] == approx(3.0) and st["backward"]["cuda_ms"] == approx(2.5)  # incl. NCCL
    assert st["head"]["cpu_ms"] == approx(1.0) and st["head"]["cuda_ms"] == approx(1.0)
    assert st["head_targets"]["cuda_ms"] == approx(0.0)
    assert st["head_convs (derived)"]["cpu_ms"] == approx(0.5)
    assert st["head_convs (derived)"]["cuda_ms"] == approx(1.0)
    assert s["device_ms_outside_stages_per_step"] == approx(0.5)  # the memcpy launched before any stage


def test_summary_no_steps():
    assert step_audit.summarize_trace([_x("audit/forward", 0, 10)])["steps"] == 0


def test_profile_end_to_end_cpu(tmp_path, monkeypatch):
    """YOLOX_PROFILE on a real CPU loop: trace exported, summary has every stage (CPU only, so no GPU time)."""
    monkeypatch.setenv("YOLOX_PROFILE", str(tmp_path / "prof"))
    monkeypatch.setenv("YOLOX_AUDIT_ITERS", "8")
    monkeypatch.setenv("YOLOX_AUDIT_OUT", str(tmp_path / "audit"))
    t = make_trainer(tmp_path, max_iter=20, max_epoch=1)
    t.audit.prof_wait, t.audit.prof_warmup, t.audit.prof_active = 2, 1, 3
    t.train_in_epoch()
    t.after_train()
    s = json.loads((tmp_path / "prof" / "summary.json").read_text())
    assert (tmp_path / "prof" / "trace_rank0.json").exists() and (tmp_path / "prof" / "key_averages.txt").exists()
    assert s["steps"] == 3 and s["step_wall_ms"] > 0
    for k in ("data_fetch", "preprocess", "forward", "backbone_neck", "head", "head_targets", "head_loss",
              "backward", "optimizer_step", "lr_update", "meters", "log"):
        assert s["stages"][k]["cpu_ms"] > 0, k
    assert s["stages"]["forward"]["cpu_ms"] >= s["stages"]["head"]["cpu_ms"]
    assert s["gpu_busy_fraction"] == 0.0
    assert step_audit._MODE == 0  # reset at finish
    assert step_audit.summarize_trace_file(str(tmp_path / "prof" / "trace_rank0.json"))["steps"] == 3


# --------------------------------------------------------------------------------------------- sync recorder
def _sync_here():
    warnings.warn("called a synchronizing CUDA operation", UserWarning)


def test_sync_recorder_captures_through_ignore_filters(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_sync_debug_mode", lambda mode: None)
    rec = step_audit.SyncRecorder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # an existing filter that would hide them
        rec.start()
        step_audit._MODE = 1
        for _ in range(2):
            rec.next_iter()
            for _ in range(3):  # same line: the default per-location registry would keep only the first
                with step_audit.stage("head"), step_audit.stage("head_targets"):
                    _sync_here()
            warnings.warn("something else")
        rec.stop()
    res = rec.result({"x": 1})
    assert res["meta"]["iters_audited"] == 2 and res["meta"]["x"] == 1
    assert res["total_syncs_per_iter"] == 3.0
    (site, info), = res["sites"].items()
    assert site.startswith("tests/perf/test_step_audit.py:") and "_sync_here" in site
    assert info["count"] == 6 and info["count_per_iter"] == 3.0 and info["iters_seen"] == [0, 1]
    assert info["stage"] == "head/head_targets"
    assert info["stack"][0].split(" ")[1].startswith("_sync_here")
    assert "test_sync_recorder_captures_through_ignore_filters" in info["stack"][1]
    assert sum(res["other_warnings"].values()) == 2
