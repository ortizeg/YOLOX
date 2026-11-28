"""Perf flag ``cuda_graphs`` (N7): replay the static-shape part of the training step as CUDA graphs.

The trunk = backbone + neck + the head's per-level convs and predictions, ending in each level's
``cat([reg, obj, cls], 1)``. For a fixed input shape it has fixed shapes and no host sync, so its
forward and backward are captured once per training size with ``torch.cuda.make_graphed_callables`` and replayed;
decode, SimOTA and the losses stay eager. One graph per multiscale size (2 * multiscale_range + 1), all sharing one
memory pool: at most one size runs per iteration and its backward always follows its own forward, so the graphs'
activations never need to coexist.

The graphs live in a registry keyed by the model object, not on the model: EMA's deepcopy, state_dict and
checkpoints see the plain model. Any input the registry does not cover (another shape, eval mode, a model that was
not captured) takes the eager path, which is the original code.

Capture runs before DDP wraps the model and before the data loader starts (no other CUDA work in flight), after
channels_last and checkpoint loading (graphs hold the parameters' storage; later in-place updates by the optimizer,
EMA reads and load_state_dict copies are seen by every replay). The warm-up passes ``make_graphed_callables`` runs
update BatchNorm running statistics; they are restored afterwards, so capture leaves the model state unchanged.
"""

from __future__ import annotations

import time
import weakref

import torch
import torch.nn as nn
from loguru import logger

# id(model) -> (weakref to model, {input shape tuple: graphed trunk})
_REGISTRY: dict[int, tuple[weakref.ref, dict[tuple, nn.Module]]] = {}


class _Trunk(nn.Module):
    """backbone + head convs as one module (shares the model's parameters; never a child of the model)."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, x):
        return tuple(self.head.raw_outputs(self.backbone(x)))


def train_sizes(exp) -> list[tuple[int, int]]:
    """Every (h, w) the trainer can feed: exp.input_size and the multiscale draws of exp.random_resize."""
    h0, w0 = exp.input_size
    factor = w0 / h0
    if hasattr(exp, "random_size") and exp.random_size is not None:
        lo, hi = exp.random_size
    else:
        r = getattr(exp, "multiscale_range", 0)
        lo, hi = int(h0 / 32) - r, int(h0 / 32) + r
    sizes = {(int(h0), int(w0))}
    for s in range(lo, hi + 1):
        sizes.add((int(32 * s), 32 * int(s * factor)))
    return sorted(sizes)


def lookup(model, x):
    """The graphed trunk for this model and input, or None (eager)."""
    entry = _REGISTRY.get(id(model))
    if entry is None or entry[0]() is not model or not model.training:
        return None
    return entry[1].get((tuple(x.shape), x.dtype, x.is_contiguous(memory_format=torch.channels_last)))


def release(model):
    _REGISTRY.pop(id(model), None)


def capture(model, sizes, batch, dtype, channels_last, amp_dtype=None, warmup=3):
    """Capture forward + backward of the trunk for each (h, w) in ``sizes`` at ``batch`` images of ``dtype``."""
    assert torch.cuda.is_available() and model.training
    dev = next(model.parameters()).device
    bn = {k: v.detach().clone() for k, v in model.state_dict().items()
          if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    graphs = {}
    t0 = time.time()
    mem0 = torch.cuda.memory_reserved(dev)
    sizes = sorted(sizes, key=lambda hw: hw[0] * hw[1], reverse=True)  # largest first: later captures fit its blocks
    params = [p for p in model.parameters() if p.requires_grad]

    def sample(h, w):
        x = torch.randn(batch, 3, h, w, device=dev).to(dtype)
        return x.contiguous(memory_format=torch.channels_last) if channels_last else x

    autocast = dict(device_type="cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.float16,
                    cache_enabled=False)
    # 1) eager warm-up of every size (cuDNN autotuning, lazy library init) on a side stream, as make_graphed_callables
    #    would do per capture; done for all sizes first because each capture's private-pool memory stays committed,
    #    and an eager warm-up after it would need a second copy of the activations
    side = torch.cuda.Stream(dev)
    side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        for h, w in sizes:
            x = sample(h, w)
            for _ in range(warmup):
                with torch.amp.autocast(**autocast):
                    outs = _Trunk(model)(x)
                torch.autograd.grad(outs, params, grad_outputs=[torch.ones_like(o) for o in outs])
                del outs
    torch.cuda.current_stream(dev).wait_stream(side)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()  # the warm-up's cached blocks (normal pool), so the graph pool can grow into them
    # 2) capture, all sharing one private pool
    pool = torch.cuda.graph_pool_handle()
    for h, w in sizes:
        x = sample(h, w)
        with torch.amp.autocast(**autocast):
            trunk = torch.cuda.make_graphed_callables(_Trunk(model), (x,), num_warmup_iters=0, pool=pool)
        graphs[(tuple(x.shape), x.dtype, channels_last)] = trunk
    torch.cuda.synchronize(dev)
    with torch.no_grad():  # undo the warm-up passes' BatchNorm statistic updates (in place: storage unchanged)
        sd = model.state_dict()
        for k, v in bn.items():
            sd[k].copy_(v)
    _REGISTRY[id(model)] = (weakref.ref(model), graphs)
    logger.info(f"perf cuda_graphs: captured {len(graphs)} trunk graphs (batch {batch}, sizes "
                f"{[s[0] for s in sizes]}) in {time.time() - t0:.1f} s, "
                f"+{(torch.cuda.memory_reserved(dev) - mem0) / 2**30:.2f} GiB reserved")
    return graphs
