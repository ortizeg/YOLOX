"""GPU-step perf flags on CPU: batched SimOTA (tier A: identical losses and gradients), class-cost algebra
(tier B: allclose), SPP chain (A), Focus as strided conv (B), foreach EMA (A)."""

import copy

import pytest
import torch

from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.models.network_blocks import Focus, SPPBottleneck
from yolox.utils import ModelEMA

from .conftest import flags


def make_labels(B, M, C, size, counts, seed):
    g = torch.Generator().manual_seed(seed)
    lab = torch.zeros(B, M, 5)
    for b, n in enumerate(counts):
        for i in range(n):
            w = float(torch.rand(1, generator=g)) * size * 0.4 + 4
            h = float(torch.rand(1, generator=g)) * size * 0.4 + 4
            cx = float(torch.rand(1, generator=g)) * (size - w) + w / 2
            cy = float(torch.rand(1, generator=g)) * (size - h) + h / 2
            lab[b, i] = torch.tensor([float(torch.randint(0, C, (1,), generator=g)), cx, cy, w, h])
    return lab


def head_losses(spec, C, counts, use_l1, seed, size=320, crowd=False):
    torch.manual_seed(seed)
    head = YOLOXHead(C, width=0.25, in_channels=[256, 512, 1024])
    head.initialize_biases(1e-2)
    head.use_l1 = use_l1
    head.train()
    B = len(counts)
    xin = [torch.randn(B, int(c * 0.25), size // s, size // s, requires_grad=True)
           for c, s in zip([256, 512, 1024], [8, 16, 32])]
    labels = make_labels(B, 120, C, size, counts, seed)
    if crowd:  # duplicate boxes -> anchors matched by several gts (conflict path) and cost ties
        labels[0, counts[0] // 2:counts[0]] = labels[0, :counts[0] - counts[0] // 2]
    with flags(spec):
        out = head(xin, labels=labels, imgs=None)
        out[0].backward()
    grads = [x.grad.clone() for x in xin] + [p.grad.clone() for p in head.parameters()]
    return [o if torch.is_tensor(o) else torch.tensor(float(o)) for o in out], grads


CASES = [(5, [7, 0, 23, 1], False, 0), (5, [40, 12], True, 1), (80, [15, 30, 2], False, 2),
         (5, [60, 60], True, 3)]


@pytest.mark.parametrize("C,counts,use_l1,seed", CASES)
def test_simota_batch_identical(C, counts, use_l1, seed):
    ref, rg = head_losses("", C, counts, use_l1, seed)
    new, ng = head_losses("simota_batch", C, counts, use_l1, seed)
    for a, b in zip(ref, new):
        assert torch.equal(a, b), (a, b)
    for a, b in zip(rg, ng):
        assert torch.equal(a, b)


def test_simota_batch_identical_crowded():
    ref, rg = head_losses("", 5, [30, 8], True, 11, crowd=True)
    new, ng = head_losses("simota_batch", 5, [30, 8], True, 11, crowd=True)
    assert all(torch.equal(a, b) for a, b in zip(ref, new))
    assert all(torch.equal(a, b) for a, b in zip(rg, ng))


@pytest.mark.parametrize("C,counts,use_l1,seed", CASES)
def test_cls_cost_algebra_close(C, counts, use_l1, seed):
    ref, rg = head_losses("", C, counts, use_l1, seed)
    new, ng = head_losses("simota_batch,cls_cost_algebra", C, counts, use_l1, seed)
    for a, b in zip(ref, new):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (a, b)
    for a, b in zip(rg, ng):
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-6)


def test_no_empty_cache_identical():
    ref, _ = head_losses("", 5, [9, 3], False, 4)
    new, _ = head_losses("no_empty_cache", 5, [9, 3], False, 4)
    assert all(torch.equal(a, b) for a, b in zip(ref, new))


@pytest.mark.parametrize("train", [True, False])
def test_sppf_identical(train):
    torch.manual_seed(0)
    m = SPPBottleneck(64, 64)
    m.train(train)
    x = torch.randn(2, 64, 23, 17)
    with flags(""):
        a = m(x)
    with flags("sppf"):
        b = m(x)
    assert torch.equal(a, b)


def test_focus_conv_close():
    torch.manual_seed(0)
    m = Focus(3, 48, ksize=3)
    m.eval()
    x = (torch.rand(2, 3, 64, 96) * 255).requires_grad_()
    with flags(""):
        a = m(x)
        a.square().mean().backward()
    ga, gw = x.grad.clone(), m.conv.conv.weight.grad.clone()
    x.grad = None
    m.zero_grad()
    with flags("focus_conv"):
        b = m(x)
        b.square().mean().backward()
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-3)
    assert torch.allclose(x.grad, ga, rtol=1e-3, atol=1e-6)
    assert torch.allclose(m.conv.conv.weight.grad, gw, rtol=1e-3, atol=1e-4)


def test_ema_foreach_identical():
    torch.manual_seed(0)
    model = YOLOX(YOLOPAFPN(0.33, 0.25), YOLOXHead(5, 0.25))
    e1 = ModelEMA(copy.deepcopy(model), 0.9998)
    e2 = ModelEMA(copy.deepcopy(model), 0.9998)
    for step in range(5):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
            for n, bf in model.named_buffers():
                if bf.dtype.is_floating_point:
                    bf.add_(0.1)
        with flags(""):
            e1.update(model)
        with flags("ema_foreach"):
            e2.update(model)
    s1, s2 = e1.ema.state_dict(), e2.ema.state_dict()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)


def test_full_model_forward_flags():
    """Whole model in train mode: tier-A model flags identical, focus_conv close."""
    torch.manual_seed(0)
    model = YOLOX(YOLOPAFPN(0.33, 0.25), YOLOXHead(5, 0.25)).train()
    x = torch.rand(2, 3, 256, 256) * 255
    lab = make_labels(2, 50, 5, 256, [6, 3], 0)
    with flags(""):
        ref = model(x, lab)["total_loss"]
    with flags("sppf,simota_batch,no_empty_cache"):
        a = model(x, lab)["total_loss"]
    with flags("all"):
        b = model(x, lab)["total_loss"]
    assert torch.equal(ref, a)
    assert torch.allclose(ref, b, rtol=1e-3)


def test_ema_defaults_unchanged():
    import math
    torch.manual_seed(0)
    m = YOLOX(YOLOPAFPN(0.33, 0.25), YOLOXHead(5, 0.25))
    e = ModelEMA(m, 0.9998)
    for x in [1, 10, 500, 2000, 5800]:
        assert e.decay(x) == 0.9998 * (1 - math.exp(-x / 2000))
    e2 = ModelEMA(m, 0.999, tau=200)
    assert abs(e2.decay(5000) - 0.999) < 1e-9
