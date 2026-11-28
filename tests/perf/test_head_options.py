"""Head assignment / loss options for the accuracy ablation (arms C and E). Off by default (== upstream, see
test_head_reference.py); here each option on its own:

- Exp.configure_head copies the options onto the head with the head's types and reports what changed
- soft_class_cost (E) equals a brute-force per-class BCE(p, one_hot * IoU) * |t - p|^2
- soft_cls_cost (E) routes the class cost through soft_class_cost, and training stays finite
- qfl_cls (E) gives loss_cls = sum over positives of BCE(logit, t) * |t - sigmoid(logit)|^beta / num_fg
- tiny_px (C) gives tiny gts at least tiny_min_k positives, never a non-candidate, and leaves other gts alone
"""

import pytest
import torch
import torch.nn.functional as F

from yolox.exp import Exp
from yolox.models import YOLOXHead
from yolox.models.yolo_head import soft_class_cost

from .test_perf_model import make_labels


def head(C=5, seed=0, **opts):
    torch.manual_seed(seed)
    h = YOLOXHead(C, width=0.25, in_channels=[256, 512, 1024])
    h.initialize_biases(1e-2)
    for k, v in opts.items():
        setattr(h, k, v)
    return h.train()


def feats(B, size, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, int(c * 0.25), size // s, size // s, generator=g).requires_grad_()
            for c, s in zip([256, 512, 1024], [8, 16, 32])]


def capture_targets(h):
    """Wrap get_targets_batched to keep its outputs."""
    got = {}
    orig = h.get_targets_batched

    def wrapped(*a, **k):
        got["out"] = orig(*a, **k)
        got["args"] = a
        return got["out"]
    h.get_targets_batched = wrapped
    return got


def test_configure_head():
    e = Exp()
    h = head()
    assert e.configure_head(h) == {}  # defaults: nothing changes
    e.merge(["tiny_px", "12", "soft_cls_cost", "1", "qfl_cls", "0", "qfl_beta", "1.5"])
    changed = e.configure_head(h)
    assert changed == {"tiny_px": 12.0, "soft_cls_cost": True, "qfl_beta": 1.5}
    assert h.soft_cls_cost is True and h.qfl_cls is False and h.tiny_px == 12.0


def test_soft_class_cost_matches_bruteforce():
    g = torch.Generator().manual_seed(0)
    B, G, A, C = 2, 3, 7, 4
    p = torch.rand(B, A, C, generator=g) * 0.98 + 0.01
    gcls = torch.randint(0, C, (B, G), generator=g)
    ious = torch.rand(B, G, A, generator=g)
    ious[0, 0, 0] = 0.0
    ref = torch.zeros(B, G, A)
    for b in range(B):
        for j in range(G):
            for a in range(A):
                t = F.one_hot(gcls[b, j], C).float() * ious[b, j, a]
                ref[b, j, a] = (F.binary_cross_entropy(p[b, a], t, reduction="none") * (t - p[b, a]) ** 2).sum()
    torch.testing.assert_close(soft_class_cost(p, gcls, ious), ref, rtol=1e-5, atol=1e-6)


def crowd(B=2, C=5, size=320, counts=(20, 6), seed=1):
    lab = make_labels(B, 30, C, size, list(counts), seed)
    lab[0, 10:20] = lab[0, :10]
    lab[0, 10:20, 1:3] += 3.0
    return lab


def test_soft_cls_cost_used_and_finite(monkeypatch):
    """The option routes SimOTA's class cost through soft_class_cost (which differs from the hard one-hot cost), and
    training stays finite. (At initialisation the class scores are near-uniform, so assignments may not move yet.)"""
    import yolox.models.yolo_head as yh
    calls = []
    real = yh.soft_class_cost

    def spy(p, gcls, ious):
        out = real(p, gcls, ious)
        hard = real(p, gcls, torch.ones_like(ious))  # t = one_hot: the hard target, same weighting
        calls.append(not torch.allclose(out, hard))
        return out
    monkeypatch.setattr(yh, "soft_class_cost", spy)
    lab, xin = crowd(), feats(2, 320, 1)
    for soft in (False, True):
        h = head(soft_cls_cost=soft)
        h.host_label_counts = [20, 6]
        out = h(xin, labels=lab, imgs=None)
        out[0].backward()
        assert torch.isfinite(out[0])
        assert len(calls) == int(soft)
    assert calls == [True]


@pytest.mark.parametrize("beta", [2.0, 1.0])
def test_qfl_cls_loss(beta):
    lab, xin = crowd(), feats(2, 320, 2)
    h = head(qfl_cls=True, qfl_beta=beta)
    got = capture_targets(h)
    h.host_label_counts = [20, 6]
    # keep the raw class logits: output[:, :, 5:] of the decoded training output
    seen = {}
    orig_get_losses = h.get_losses

    def spy(imgs, xs, ys, st, labels, outputs, origin, dtype):
        seen["cls"] = outputs[:, :, 5:].detach().reshape(-1, h.num_classes)
        return orig_get_losses(imgs, xs, ys, st, labels, outputs, origin, dtype)
    h.get_losses = spy
    out = h(xin, labels=lab, imgs=None)
    cls_t, fg, num_fg = got["out"][0], got["out"][3], got["out"][5]
    logit = seen["cls"]
    per = F.binary_cross_entropy_with_logits(logit, cls_t, reduction="none") * (cls_t - logit.sigmoid()).abs() ** beta
    expect = (per * fg[:, None].float()).sum() / num_fg.clamp(min=1).float()
    torch.testing.assert_close(out[3], expect, rtol=1e-5, atol=1e-7)


def test_tiny_rule():
    size = 320
    lab = torch.zeros(1, 4, 5)
    lab[0, 0] = torch.tensor([1.0, 100.0, 100.0, 6.0, 6.0])  # tiny (ball-sized)
    lab[0, 1] = torch.tensor([0.0, 200.0, 150.0, 40.0, 90.0])  # player-sized
    xin = feats(1, size, 3)
    res = {}
    for tiny in (0.0, 12.0):
        h = head(tiny_px=tiny, tiny_min_k=2)
        got = capture_targets(h)
        h.host_label_counts = [2]
        h(xin, labels=lab, imgs=None)
        _, reg_t, _, fg, _, _, _ = got["out"]
        on_tiny = fg & (reg_t[:, 2] == 6.0)
        on_big = fg & (reg_t[:, 2] == 40.0)
        res[tiny] = (int(on_tiny.sum()), on_big.clone())
        # every tiny-gt positive is a centre-region candidate (|anchor centre - gt centre| < 1.5 strides)
        st, xs, ys = (t[0] for t in got["args"][5:8])
        xc, yc = (xs + 0.5) * st, (ys + 0.5) * st
        inside = ((xc - 100.0).abs() < 1.5 * st) & ((yc - 100.0).abs() < 1.5 * st)
        assert bool(inside[on_tiny].all())
    assert res[12.0][0] >= 2 and res[12.0][0] >= res[0.0][0]
    assert torch.equal(res[0.0][1], res[12.0][1])  # the large gt's assignment is unchanged


def test_obj_rel_target():
    """Each gt's best positive gets objectness target 1, its other positives IoU / best IoU < 1, background 0; the
    o2m positives themselves are unchanged."""
    lab, xin = crowd(), feats(2, 320, 5)
    res = {}
    for rel in (False, True):
        h = head(obj_rel_target=rel)
        got = capture_targets(h)
        h.host_label_counts = [20, 6]
        out = h(xin, labels=lab, imgs=None)
        out[0].backward()
        assert torch.isfinite(out[0])
        cls_t, reg_t, obj_t, fg, _, _, _ = got["out"]
        res[rel] = (fg.clone(), obj_t.clone(), reg_t.clone(), cls_t.clone())
    fg0, obj0, reg_t, cls_t = res[False]
    fg1, obj1, _, _ = res[True]
    assert torch.equal(fg0, fg1) and torch.equal(obj0[:, 0], fg0.float())
    assert float(obj1[~fg1].abs().sum()) == 0.0
    pos_iou = cls_t.max(1).values  # IoU-scaled class target = the positive's IoU with its gt
    for key in {tuple(x.tolist()) for x in reg_t[fg1]}:
        same = fg1 & (reg_t == torch.tensor(key)).all(1)
        o, iou = obj1[same, 0], pos_iou[same]
        torch.testing.assert_close(o.max(), torch.tensor(1.0))
        torch.testing.assert_close(o, iou / iou.max(), rtol=1e-5, atol=1e-6)

