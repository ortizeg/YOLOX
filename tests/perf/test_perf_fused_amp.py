"""fused_optim_amp (tier B): the exp's optimizer rebuilt with fused=True in the constructor.

CPU (torch 2.5.1 has fused SGD/Adam/AdamW kernels on CPU, so the rebuild is exercised with require_cuda=False):
same class and param groups, _step_supports_amp_scaling set, fused vs unfused updates allclose, checkpoints
interchangeable both ways, flag off = the exp's optimizer object untouched. CUDA: GradScaler.step/update with the
fused optimizer does no host sync, and fused vs unfused weights stay allclose over AMP steps."""

import copy

import pytest
import torch
from torch import nn

from yolox.exp import Exp
from yolox.utils.fused_optim import apply_flag, rebuild_fused

from .conftest import flags

RTOL, ATOL = 1e-5, 1e-6


def tiny_model(seed=0, device="cpu"):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.SiLU(), nn.Conv2d(8, 4, 1)).to(device)


def yolox_groups(model):
    """YOLOX's three groups: BN weights (no decay), conv weights (decay), biases."""
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    return pg0, pg1, pg2


def make_sgd(model):  # yolox/exp/yolox_base.py get_optimizer
    pg0, pg1, pg2 = yolox_groups(model)
    opt = torch.optim.SGD(pg0, lr=0.01, momentum=0.9, nesterov=True)
    opt.add_param_group({"params": pg1, "weight_decay": 5e-4})
    opt.add_param_group({"params": pg2})
    return opt


def make_adamw(model):  # exps/default/yolox_m_basketball_adamw.py style
    pg0, pg1, pg2 = yolox_groups(model)
    opt = torch.optim.AdamW(pg0, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.98))
    opt.add_param_group({"params": pg1, "weight_decay": 0.05})
    opt.add_param_group({"params": pg2, "weight_decay": 0.0, "lr": 2e-3})
    return opt


MAKERS = {"sgd": make_sgd, "adamw": make_adamw}


def set_grads(model, seed):
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.randn(p.shape, generator=g).to(p.device)


def run_steps(model, opt, seeds, lr_schedule=True):
    for s in seeds:
        set_grads(model, s)
        opt.step()
        if lr_schedule:  # the trainer rewrites every group's lr each iteration
            for g in opt.param_groups:
                g["lr"] = g["lr"] * 0.9
    return [p.detach().clone() for p in model.parameters()]


def assert_close(a, b):
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("kind", list(MAKERS))
def test_rebuild_keeps_class_and_groups(kind):
    model = tiny_model()
    opt = MAKERS[kind](model)
    new = rebuild_fused(opt, require_cuda=False)
    assert new is not opt and type(new) is type(opt)
    assert len(new.param_groups) == len(opt.param_groups)
    for g0, g1 in zip(opt.param_groups, new.param_groups):
        assert [id(p) for p in g0["params"]] == [id(p) for p in g1["params"]]
        assert set(g0) == set(g1)
        for k in g0:
            if k not in ("params", "fused", "foreach"):
                assert g0[k] == g1[k], k
        assert g1["fused"] is True and g1["foreach"] is False
    assert getattr(new, "_step_supports_amp_scaling", False)
    assert not getattr(opt, "_step_supports_amp_scaling", False)
    assert not new.state


def test_rebuild_yolox_exp_optimizer():
    exp = Exp()
    exp.depth, exp.width = 0.33, 0.125
    exp.get_model()
    opt = exp.get_optimizer(8)
    new = rebuild_fused(opt, require_cuda=False)
    assert type(new) is torch.optim.SGD and len(new.param_groups) == 3
    assert [g["weight_decay"] for g in new.param_groups] == [g["weight_decay"] for g in opt.param_groups]
    assert all(g["nesterov"] and g["momentum"] == exp.momentum for g in new.param_groups)


@pytest.mark.parametrize("kind", list(MAKERS))
def test_fused_step_allclose(kind):
    m0, m1 = tiny_model(), tiny_model()
    ref = run_steps(m0, MAKERS[kind](m0), range(5))
    new = run_steps(m1, rebuild_fused(MAKERS[kind](m1), require_cuda=False), range(5))
    assert_close(ref, new)


@pytest.mark.parametrize("kind", list(MAKERS))
@pytest.mark.parametrize("direction", ["unfused_to_fused", "fused_to_unfused"])
def test_resume_interchangeable(kind, direction):
    """3 steps with one optimizer, save, load into the other kind, 3 more steps == 6 unfused steps."""
    m_ref = tiny_model()
    ref = run_steps(m_ref, MAKERS[kind](m_ref), range(6))

    m_a = tiny_model()
    opt_a = MAKERS[kind](m_a)
    if direction == "fused_to_unfused":
        opt_a = rebuild_fused(opt_a, require_cuda=False)
    run_steps(m_a, opt_a, range(3))
    sd = copy.deepcopy(opt_a.state_dict())

    m_b = copy.deepcopy(m_a)
    opt_b = MAKERS[kind](m_b)
    if direction == "unfused_to_fused":
        opt_b = rebuild_fused(opt_b, require_cuda=False)
    opt_b.load_state_dict(sd)
    for g in opt_b.param_groups:
        if direction == "unfused_to_fused":
            assert g["fused"] is True and g["foreach"] is False
        else:  # the fused run's checkpoint is written in the original format
            assert not g["fused"]
    if kind == "adamw":
        steps = [st["step"] for st in opt_b.state.values()]
        assert all(s.device.type == "cpu" and s.dtype == torch.float32 for s in steps)
    out = run_steps(m_b, opt_b, range(3, 6))
    assert_close(ref, out)


def test_saved_state_dict_does_not_touch_live_state():
    m = tiny_model()
    opt = rebuild_fused(make_adamw(m), require_cuda=False)
    run_steps(m, opt, range(2))
    live = {p: (st, st["step"]) for p, st in opt.state.items()}
    sd = opt.state_dict()
    assert all(not g["fused"] for g in sd["param_groups"])
    assert all(g["fused"] is True and g["foreach"] is False for g in opt.param_groups)
    for p, (st, step) in live.items():
        assert opt.state[p] is st and st["step"] is step


def test_existing_state_carried_over():
    m0, m1 = tiny_model(), tiny_model()
    ref = run_steps(m0, make_sgd(m0), range(4))
    opt = make_sgd(m1)
    run_steps(m1, opt, range(2))
    opt = rebuild_fused(opt, require_cuda=False)
    assert all(g["fused"] for g in opt.param_groups)
    assert_close(ref, run_steps(m1, opt, range(2, 4)))


def test_flag_off_untouched():
    exp = Exp()
    exp.depth, exp.width = 0.33, 0.125
    exp.get_model()
    opt = exp.get_optimizer(8)
    groups = copy.deepcopy([{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups])
    with flags(""):
        out = apply_flag(opt, exp, require_cuda=False)
    assert out is opt and exp.optimizer is opt
    assert [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups] == groups
    assert not getattr(opt, "_step_supports_amp_scaling", False)


def test_flag_on_cpu_params_keep_original():
    """The trainer requires CUDA params; on CPU the original optimizer is kept (and logged)."""
    m = tiny_model()
    opt = make_sgd(m)
    with flags("fused_optim_amp"):
        assert apply_flag(opt) is opt


def test_flag_on_replaces_exp_cache():
    exp = Exp()
    exp.depth, exp.width = 0.33, 0.125
    exp.get_model()
    opt = exp.get_optimizer(8)
    with flags("fused_optim_amp"):
        new = apply_flag(opt, exp, require_cuda=False)
    assert new is not opt and exp.get_optimizer(8) is new


def test_unsupported_class_kept():
    m = tiny_model()
    opt = torch.optim.RMSprop(m.parameters(), lr=0.01)
    assert rebuild_fused(opt, require_cuda=False) is opt


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def amp_steps(model, opt, n, seed=0):
    scaler = torch.amp.GradScaler("cuda")
    torch.manual_seed(seed)
    xs = [torch.randn(4, 3, 32, 32, device="cuda") for _ in range(n)]
    for x in xs:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = model(x).float().pow(2).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return scaler, [p.detach().clone() for p in model.parameters()]


@cuda
@pytest.mark.parametrize("kind", list(MAKERS))
def test_cuda_no_host_sync_in_scaler_step(kind):
    m = tiny_model(device="cuda")
    opt = rebuild_fused(MAKERS[kind](m))
    assert opt.param_groups[0]["fused"] and opt._step_supports_amp_scaling
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(4, 3, 32, 32, device="cuda")
    for _ in range(3):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = m(x).float().pow(2).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            scaler.step(opt)
            scaler.update()
        finally:
            torch.cuda.set_sync_debug_mode("default")


@cuda
def test_cuda_unfused_scaler_step_does_sync():
    """Shows the check above is meaningful: the generic GradScaler path syncs (found_inf.item())."""
    m = tiny_model(device="cuda")
    opt = make_sgd(m)
    scaler = torch.amp.GradScaler("cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = m(torch.randn(4, 3, 32, 32, device="cuda")).float().pow(2).mean()
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with pytest.raises(RuntimeError):
            scaler.step(opt)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@cuda
@pytest.mark.parametrize("kind", list(MAKERS))
def test_cuda_amp_fused_allclose(kind):
    m0, m1 = tiny_model(device="cuda"), tiny_model(device="cuda")
    s0, ref = amp_steps(m0, MAKERS[kind](m0), 8)
    s1, new = amp_steps(m1, rebuild_fused(MAKERS[kind](m1)), 8)
    assert s0.get_scale() == s1.get_scale()
    for x, y in zip(ref, new):
        torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)
