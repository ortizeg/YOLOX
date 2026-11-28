"""Arm A: NMS-free dual head (one-to-many + one-to-one branches on shared towers, ProgLoss).

- one-to-one targets: at most one positive per gt, every positive anchor unique and a candidate of its gt
- the o2m branch of a dual head assigns and scores exactly like a single head with the same weights; the o2o branch
  like a single head whose prediction layers are the o2o layers, with one_to_one=True
- the training loss is the ProgLoss blend of those two losses; gradients reach both branches and the shared towers
- inference returns the chosen branch's decoded output (== a single head with that branch's layers)
- o2o initialisation from o2m happens once and survives a state_dict round trip; Exp.configure_head enables it
- CUDA (skipped elsewhere): CUDA-graph capture of the dual trunk trains like eager
"""

import copy

import pytest
import torch

from yolox.exp import Exp
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

from .test_perf_model import make_labels

TOL = dict(rtol=2e-5, atol=1e-6)


def dual(C=5, seed=0, use_l1=False):
    torch.manual_seed(seed)
    h = YOLOXHead(C, width=0.25, in_channels=[256, 512, 1024])
    h.initialize_biases(1e-2)
    h.enable_dual_head()
    torch.manual_seed(seed + 100)
    with torch.no_grad():  # distinct o2o weights, so the branches are distinguishable
        for m in (h.cls_preds_o2o, h.reg_preds_o2o, h.obj_preds_o2o):
            for p in m.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    h.o2o_initialized.fill_(1)
    h.use_l1 = use_l1
    return h.train()


def single_from(h, branch):
    """A single head with the dual head's towers and one branch's prediction layers."""
    s = YOLOXHead(h.num_classes, width=0.25, in_channels=[256, 512, 1024])
    sd = {k: v for k, v in h.state_dict().items() if "_o2o" not in k and k != "o2o_initialized"}
    if branch == "o2o":
        for name in ("cls_preds", "reg_preds", "obj_preds"):
            for k, v in getattr(h, name + "_o2o").state_dict().items():
                sd[f"{name}.{k}"] = v
    s.load_state_dict(sd)
    s.use_l1 = h.use_l1
    return s.train(h.training)


def feats(B, size, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, int(c * 0.25), size // s, size // s, generator=g).requires_grad_()
            for c, s in zip([256, 512, 1024], [8, 16, 32])]


def capture(h):
    calls = []
    orig = h.get_targets_batched

    def wrapped(*a, **k):
        out = orig(*a, **k)
        calls.append((a, k, out))
        return out
    h.get_targets_batched = wrapped
    return calls


@pytest.mark.parametrize("use_l1", [False, True])
def test_one_to_one_targets(use_l1):
    counts = [12, 0, 5]
    labels = make_labels(3, 20, 5, 320, counts, 1)
    labels[0, 6:12] = labels[0, :6]  # duplicates: two gts competing for the same cheapest anchor
    labels[0, 6:12, 1:3] += 2.0
    h = dual(use_l1=use_l1)
    calls = capture(h)
    h.host_label_counts = counts
    out = h(feats(3, 320, 1), labels=labels, imgs=None)
    out[0].backward()
    assert [c[1]["one_to_one"] for c in calls] == [False, True]
    (a, _, (_, reg_t, _, fg, _, num_fg, _)) = calls[1]
    A = a[2].shape[1]
    fg = fg.view(3, A)
    assert int(num_fg) == int(fg.sum()) <= sum(counts)
    assert int(fg[1].sum()) == 0  # image without labels
    st, xs, ys = (t[0] for t in a[5:8])
    xc, yc = (xs + 0.5) * st, (ys + 0.5) * st
    reg = reg_t.view(3, A, 4)
    for b in range(3):
        pos = torch.nonzero(fg[b]).squeeze(1)
        boxes = reg[b, pos]
        # unique gt per positive, and each positive is a centre-region candidate of its gt
        assert len({tuple(x.tolist()) for x in boxes}) == len(pos)
        inside = ((xc[pos] - boxes[:, 0]).abs() < 1.5 * st[pos]) & ((yc[pos] - boxes[:, 1]).abs() < 1.5 * st[pos])
        assert bool(inside.all())


@pytest.mark.parametrize("progress", [0.0, 0.37, 1.0])
def test_loss_is_progloss_blend_of_single_heads(progress):
    counts = [7, 3]
    labels = make_labels(2, 20, 5, 320, counts, 2)
    xin = feats(2, 320, 2)
    h = dual(use_l1=True)
    h.progress = progress
    h.host_label_counts = counts
    out = h(xin, labels=labels, imgs=None)
    m, o = single_from(h, "o2m"), single_from(h, "o2o")
    m.host_label_counts = counts
    lm = m(xin, labels=labels, imgs=None)
    calls = capture(o)
    orig = o.get_losses
    o.get_losses = lambda *a, **k: orig(*a, **{**k, "one_to_one": True})
    o.host_label_counts = counts
    lo = o(xin, labels=labels, imgs=None)
    wm, wo = h.progloss_weights()
    assert (wm, wo) == pytest.approx((0.8 - 0.7 * progress, 0.2 + 0.7 * progress))
    for i in range(5):
        torch.testing.assert_close(torch.as_tensor(out[i]).float(),
                                   (wm * torch.as_tensor(lm[i]) + wo * torch.as_tensor(lo[i])).float(), **TOL)
    torch.testing.assert_close(torch.as_tensor(out[5]).float(), torch.as_tensor(lm[5]).float(), **TOL)
    assert calls[0][1]["one_to_one"] is True


def test_gradients_reach_both_branches_and_towers():
    h = dual()
    h.host_label_counts = [6, 2]
    out = h(feats(2, 256, 3), labels=make_labels(2, 20, 5, 256, [6, 2], 3), imgs=None)
    out[0].backward()
    for name in ("cls_preds", "reg_preds", "obj_preds", "cls_preds_o2o", "reg_preds_o2o", "obj_preds_o2o",
                 "cls_convs", "reg_convs", "stems"):
        g = [p.grad for p in getattr(h, name).parameters()]
        assert all(x is not None and torch.isfinite(x).all() for x in g), name
        assert any(float(x.abs().sum()) > 0 for x in g), name


@pytest.mark.parametrize("branch", ["o2o", "o2m"])
def test_inference_uses_the_chosen_branch(branch):
    h = dual().eval()
    h.infer_o2o = branch == "o2o"
    s = single_from(h, branch).eval()
    xin = [x.detach() for x in feats(2, 320, 4)]
    with torch.no_grad():  # equal up to float rounding (sigmoid over a channel slice vs a contiguous tensor)
        torch.testing.assert_close(h(xin), s(xin), rtol=1e-6, atol=1e-8)


def test_o2o_init_once_and_round_trip():
    torch.manual_seed(0)
    h = YOLOXHead(5, width=0.25, in_channels=[256, 512, 1024])
    h.enable_dual_head()
    with torch.no_grad():
        h.cls_preds[0].weight.add_(1.0)  # "loaded" o2m weights differ from the init copy
    assert h.init_o2o_from_o2m() is True
    assert torch.equal(h.cls_preds_o2o[0].weight, h.cls_preds[0].weight)
    with torch.no_grad():
        h.cls_preds_o2o[0].weight.add_(0.5)  # "trained" o2o
    assert h.init_o2o_from_o2m() is False  # never overwrites a trained/loaded o2o branch
    sd = copy.deepcopy(h.state_dict())
    assert "o2o_initialized" in sd and any("cls_preds_o2o" in k for k in sd)
    h2 = YOLOXHead(5, width=0.25, in_channels=[256, 512, 1024])
    h2.enable_dual_head()
    h2.load_state_dict(sd)
    assert h2.init_o2o_from_o2m() is False
    assert torch.equal(h2.cls_preds_o2o[0].weight, h.cls_preds_o2o[0].weight)


def test_configure_head_enables_dual():
    e = Exp()
    e.merge(["dual_head", "1", "infer_o2o", "0", "progloss", "0.5,0.5,0.5,0.5"])
    model = YOLOX(YOLOPAFPN(0.33, 0.25), YOLOXHead(5, 0.25))
    changed = e.configure_head(model.head)
    assert changed["dual_head"] is True and model.head.dual_head and model.head.infer_o2o is False
    assert model.head.progloss == (0.5, 0.5, 0.5, 0.5)
    assert hasattr(model.head, "cls_preds_o2o")
    assert e.configure_head(model.head) == {}  # idempotent


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dual_head_cuda_graphs_match_eager():
    from yolox.utils import cuda_graphs

    from .conftest import flags
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    sizes, B = [(256, 256)], 2

    def run(capture):
        torch.manual_seed(0)
        m = YOLOX(YOLOPAFPN(0.33, 0.25), YOLOXHead(5, 0.25)).cuda().train()
        m.head.enable_dual_head()
        m.head.init_o2o_from_o2m()
        opt = torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9)
        losses = []
        with flags("cuda_graphs" if capture else ""):
            if capture:
                cuda_graphs.capture(m, sizes, B, torch.float32, False, None)
            for i in range(3):
                x = torch.randn(B, 3, 256, 256, generator=torch.Generator().manual_seed(i)).cuda()
                m.head.host_label_counts = [4, 2]
                m.head.progress = i / 3
                assert (cuda_graphs.lookup(m, x) is not None) == capture
                out = m(x, make_labels(B, 20, 5, 256, [4, 2], i).cuda())
                opt.zero_grad(set_to_none=True)
                out["total_loss"].backward()
                opt.step()
                losses.append(out["total_loss"].detach())
        cuda_graphs.release(m)
        return losses
    for a, b in zip(run(False), run(True)):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
