"""``fused_optim_amp`` perf flag: rebuild the exp's optimizer with ``fused=True`` passed to the constructor.

Why the constructor matters (torch 2.5.1): ``SGD/Adam/AdamW.__init__`` set ``optimizer._step_supports_amp_scaling``
only when ``fused=True`` is a constructor argument. ``GradScaler.step`` checks that attribute: with it, the fused
kernel unscales the grads and skips the update on inf/nan from device tensors; without it, GradScaler runs its own
unscale pass and calls ``found_inf.item()`` (a host sync) every step. Setting ``group["fused"] = True`` after
construction (the ``fused_optim`` flag) gets the fused kernel but not the sync-free AMP path, and only for Adam(W).

The rebuilt optimizer has the same class and the same param groups (every per-group hyperparameter kept) with
``fused=True, foreach=False``. Two hooks keep checkpoints interchangeable with the original optimizer:

- load: ``Optimizer.load_state_dict`` replaces the param groups with the saved ones (so a checkpoint from an unfused
  run would silently switch ``fused`` back off while ``_step_supports_amp_scaling`` stays on) and keeps Adam's
  ``step`` on the CPU when the saved group was not fused. The post-load hook re-applies ``fused``/``foreach`` and
  moves ``step`` to the parameter's device as float32, as the fused kernel requires.
- SGD first step (torch 2.5.1 bug, ``torch/optim/sgd.py`` ``_fused_sgd``): on the first step the momentum buffers
  are ``torch.empty_like(grad)`` and the kernel is told ``is_first_step`` to fill them. When GradScaler finds inf/nan
  (routine at the initial 65536 scale) the kernel skips the update, but ``SGD.step`` still stores the uninitialized
  buffers; the next step treats them as real momentum and the weights go NaN. A step pre-hook creates every missing
  buffer as zeros first. With ``dampening == 0`` (the only case fused SGD is used for here) ``momentum * 0 + grad``
  is exactly ``grad``, i.e. what the unfused first step stores, and a skipped step leaves the zeros untouched.
- save: the state dict is written with each group's original ``fused``/``foreach`` values and Adam's ``step`` back on
  the CPU when the original group was neither fused nor capturable, so resuming such a checkpoint without the flag
  gives exactly the original optimizer.
"""

from __future__ import annotations

import inspect

import torch
from loguru import logger

FUSABLE = (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)


def why_not_fusable(optimizer: torch.optim.Optimizer, require_cuda: bool = True) -> str | None:
    """None when ``optimizer`` can be rebuilt fused, else the reason it is kept as is."""
    if type(optimizer) not in FUSABLE:  # exact type: a subclass may take different constructor arguments
        return f"{type(optimizer).__name__} has no fused kernel handled here (SGD, Adam, AdamW)"
    for g in optimizer.param_groups:
        if type(optimizer) is torch.optim.SGD and g["momentum"] != 0 and g["dampening"] != 0:
            return "SGD with dampening != 0 (zero-initialised momentum would change its first step)"
        if g.get("differentiable", False):
            return "a param group has differentiable=True (fused does not support it)"
        for p in g["params"]:
            if not torch.is_floating_point(p):
                return f"non floating point param ({p.dtype})"
            if require_cuda and p.device.type != "cuda":
                return f"params are on {p.device.type}, not cuda"
            if p.is_sparse:
                return "sparse param"
    return None


def _zero_missing_momentum(optimizer: torch.optim.Optimizer, args=None, kwargs=None) -> None:
    """SGD step pre-hook: zero momentum buffers for params about to be stepped that have none yet (see module doc).
    Scans only until every param has a buffer; a state load re-arms it."""
    if getattr(optimizer, "_yolox_bufs_ready", False):
        return
    ready = True
    for g in optimizer.param_groups:
        if g["momentum"] == 0:
            continue
        for p in g["params"]:
            st = optimizer.state[p]
            if st.get("momentum_buffer") is None:
                if p.grad is None:
                    ready = False
                else:
                    st["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
    optimizer._yolox_bufs_ready = ready


def _keep_fused(optimizer: torch.optim.Optimizer) -> None:
    """Post-load hook: the loaded param groups come from the checkpoint; make them fused again."""
    optimizer._yolox_bufs_ready = False
    for g in optimizer.param_groups:
        g["fused"], g["foreach"] = True, False
        for p in g["params"]:
            st = optimizer.state.get(p)
            if st and torch.is_tensor(st.get("step")):
                st["step"] = st["step"].to(dtype=torch.float32, device=p.device)


def _make_save_hook(orig_groups):
    def _as_original(optimizer, state_dict):
        """State-dict post hook: write the groups' original fused/foreach and host Adam's step as the original
        optimizer would. Inner state dicts are copied, never modified (they are the live optimizer state)."""
        ids_cpu_step = set()
        for g, (fused, foreach, capturable) in zip(state_dict["param_groups"], orig_groups):
            g["fused"], g["foreach"] = fused, foreach
            if not (fused or capturable):
                ids_cpu_step.update(g["params"])
        state = {}
        for k, v in state_dict["state"].items():
            if k in ids_cpu_step and torch.is_tensor(v.get("step")):
                v = dict(v, step=v["step"].to("cpu"))
            state[k] = v
        state_dict["state"] = state
        return state_dict

    return _as_original


def rebuild_fused(optimizer: torch.optim.Optimizer, require_cuda: bool = True) -> torch.optim.Optimizer:
    """Return a fused copy of ``optimizer`` (same class, same param groups), or ``optimizer`` itself (with a log
    line) when it cannot be fused. Call before any state exists or a checkpoint is loaded; existing state is
    carried over through ``load_state_dict`` if there is any."""
    reason = why_not_fusable(optimizer, require_cuda)
    if reason is not None:
        logger.warning(f"fused_optim_amp: keeping the original {type(optimizer).__name__}: {reason}")
        return optimizer
    orig_groups = [(g.get("fused"), g.get("foreach"), g.get("capturable", False)) for g in optimizer.param_groups]
    groups = [{**g, "params": list(g["params"]), "fused": True, "foreach": False} for g in optimizer.param_groups]
    defaults = {**optimizer.defaults, "fused": True, "foreach": False}
    # only what the constructor takes: torch >= 2.10 AdamW keeps `decoupled_weight_decay` in its defaults (it shares
    # Adam's implementation) but AdamW.__init__ does not accept it
    params = inspect.signature(type(optimizer).__init__).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        defaults = {k: v for k, v in defaults.items() if k in params}
    new = type(optimizer)(groups, **defaults)
    new.register_load_state_dict_post_hook(_keep_fused)
    if type(new) is torch.optim.SGD:
        new.register_step_pre_hook(_zero_missing_momentum)
    if optimizer.state:
        new.load_state_dict(optimizer.state_dict())
    new.register_state_dict_post_hook(_make_save_hook(orig_groups))
    logger.info(f"fused_optim_amp: rebuilt {type(optimizer).__name__} with fused=True "
                f"({len(groups)} param groups, AMP unscale/inf-check inside the kernel)")
    return new


def apply_flag(optimizer: torch.optim.Optimizer, exp=None, require_cuda: bool = True) -> torch.optim.Optimizer:
    """Trainer entry point: with ``fused_optim_amp`` off, return ``optimizer`` untouched. With it on, return the
    fused rebuild and point the exp's cached ``optimizer`` (``Exp.get_optimizer`` memoizes it) at it."""
    from yolox.utils import perf

    if not perf.on("fused_optim_amp"):
        return optimizer
    new = rebuild_fused(optimizer, require_cuda)
    if exp is not None and exp.__dict__.get("optimizer") is optimizer:
        exp.optimizer = new
    return new
