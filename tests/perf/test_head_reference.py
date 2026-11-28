"""The production YOLOX head against the frozen pre-perf head (tests/perf/reference/, fork commit 4488863).

The training path is now one path: batched SimOTA, masked dense losses, device-side grids, w*h IoU areas (the former
flags simota_batch, cls_cost_algebra, simota_nosync, head_nosync, iou_nosync, dense_loss, no_empty_cache). Each was
validated against the original code on its own when it was a flag; these tests check the combined path against the
original directly:

- training losses, num_fg/num_gts and every gradient match (tier B: summation order and the class-cost algebra),
  over class counts, images without labels, a batch without labels, crowded boxes, the L1 loss, several input
  sizes on one head object (grid cache), and with or without host label counts
- inference decode is bit-identical
- IOUloss forward is bit-identical and its gradient matches (iou and giou)
- the full YOLOX model gives the same training outputs with either head
"""

import pytest
import torch

from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.models.losses import IOUloss

from .reference.upstream_head import UpstreamYOLOXHead
from .reference.upstream_losses import IOUloss as UpstreamIOUloss
from .test_perf_model import make_labels

LOSS_TOL = dict(rtol=2e-5, atol=1e-6)
GRAD_TOL = dict(rtol=1e-4, atol=1e-6)


def heads(C, use_l1, seed):
    torch.manual_seed(seed)
    new = YOLOXHead(C, width=0.25, in_channels=[256, 512, 1024])
    new.initialize_biases(1e-2)
    ref = UpstreamYOLOXHead(C, width=0.25, in_channels=[256, 512, 1024])
    ref.load_state_dict(new.state_dict())
    for h in (new, ref):
        h.use_l1 = use_l1
        h.train()
    return new, ref


def feats(B, size, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, int(c * 0.25), size // s, size // s, generator=g).requires_grad_()
            for c, s in zip([256, 512, 1024], [8, 16, 32])]


def crowd_labels(B, M, C, size, counts, seed):
    """Boxes stacked on a few centres (heavy overlap, shared candidates, many disputed anchors)."""
    lab = make_labels(B, M, C, size, counts, seed)
    for b, n in enumerate(counts):
        for i in range(n):
            lab[b, i, 1:3] = lab[b, i % 3, 1:3] + (i % 5)
    return lab


def step(head, xin, labels, counts=None):
    for x in xin:
        x.grad = None
    head.zero_grad()
    if counts is not None:
        head.host_label_counts = list(counts)
    out = head(xin, labels=labels, imgs=None)
    out[0].backward()
    outs = [torch.as_tensor(o).detach().float().reshape(()) for o in out]
    grads = [x.grad.clone() for x in xin] + [p.grad.clone() for p in head.parameters()]
    return outs, grads


def assert_match(a, b):
    (ao, ag), (bo, bg) = a, b
    for x, y in zip(ao, bo):  # total, iou, obj, cls, l1, num_fg / num_gts
        torch.testing.assert_close(x, y, **LOSS_TOL)
    for x, y in zip(ag, bg):
        torch.testing.assert_close(x, y, **GRAD_TOL)


@pytest.mark.parametrize("host_counts", [False, True])
@pytest.mark.parametrize("C,counts,use_l1,seed", [(5, [7, 0, 23, 1], False, 0), (5, [40, 12], True, 1),
                                                  (80, [15, 30, 2], False, 2), (5, [3, 0], True, 3),
                                                  (5, [0, 0], True, 4), (365, [9, 4], False, 5),
                                                  (1, [60, 1], True, 6), (5, [60, 60], True, 3)])
def test_training_matches_upstream(host_counts, C, counts, use_l1, seed):
    new, ref = heads(C, use_l1, seed)
    labels = make_labels(len(counts), 70, C, 320, counts, seed)
    xin = feats(len(counts), 320, seed)
    assert_match(step(new, xin, labels, counts if host_counts else None), step(ref, xin, labels))


@pytest.mark.parametrize("dup", [False, True])
def test_crowded_matches_upstream(dup):
    new, ref = heads(5, True, 11)
    counts = [30, 8]
    labels = crowd_labels(2, 40, 5, 320, counts, 11)
    if dup:  # exact duplicate boxes: anchors matched by several gts, cost ties
        labels[0, 15:30] = labels[0, :15]
    xin = feats(2, 320, 11)
    assert_match(step(new, xin, labels, counts), step(ref, xin, labels))


def test_several_sizes_one_head():
    """Same head objects across sizes (the device grid cache is built, reused and revisited)."""
    new, ref = heads(5, False, 7)
    sizes = [320, 256, 384, 320]
    for i, size in enumerate(sizes):
        counts = [5 + i, 2]
        labels = make_labels(2, 30, 5, size, counts, i)
        xin = feats(2, size, 100 + i)
        assert_match(step(new, xin, labels, counts), step(ref, xin, labels))
    assert len(new._nosync_cache) == 3 * 3  # 3 levels x 3 distinct sizes


def test_decode_identical():
    new, ref = heads(5, False, 3)
    new.eval()
    ref.eval()
    for size in [320, 256, 320]:
        xin = feats(2, size, size)
        with torch.no_grad():
            assert torch.equal(new(xin), ref(xin))
        # tools/eval.py and tools/demo.py call decode_outputs(outputs, dtype=outputs.type()) when decoding outside
        raw = torch.randn(2, sum((size // s) ** 2 for s in (8, 16, 32)), 10)
        new.hw = ref.hw = [(size // s, size // s) for s in (8, 16, 32)]
        assert torch.equal(new.decode_outputs(raw, dtype=raw.type()), ref.decode_outputs(raw, dtype=raw.type()))


@pytest.mark.parametrize("loss_type", ["iou", "giou"])
def test_iou_loss_matches_upstream(loss_type):
    g = torch.Generator().manual_seed(0)
    tgt = torch.cat([torch.rand(500, 2, generator=g) * 300, torch.rand(500, 2, generator=g) * 80 + 1], 1)
    pred0 = tgt + torch.randn(500, 4, generator=g) * 6
    pred0[:, 2:] = pred0[:, 2:].abs() + 0.5
    pred0[50:60] = tgt[50:60]  # identical boxes
    res = []
    for cls in (IOUloss, UpstreamIOUloss):
        pred = pred0.clone().requires_grad_()
        loss = cls(reduction="none", loss_type=loss_type)(pred, tgt)
        loss.sum().backward()
        res.append((loss.detach(), pred.grad))
    assert torch.equal(res[0][0], res[1][0])
    torch.testing.assert_close(res[0][1], res[1][1], rtol=1e-5, atol=1e-7)


def test_full_model_matches_upstream():
    torch.manual_seed(0)
    new = YOLOX(YOLOPAFPN(0.33, 0.25), YOLOXHead(5, 0.25)).train()
    torch.manual_seed(0)
    ref = YOLOX(YOLOPAFPN(0.33, 0.25), UpstreamYOLOXHead(5, 0.25)).train()
    ref.load_state_dict(new.state_dict())
    imgs = torch.randn(2, 3, 256, 256)
    labels = make_labels(2, 50, 5, 256, [6, 3], 0)
    a, b = new(imgs, labels), ref(imgs, labels)
    for k in a:
        torch.testing.assert_close(torch.as_tensor(a[k]).float(), torch.as_tensor(b[k]).float(), **LOSS_TOL)


def test_visualize_assign_result_runs(tmp_path):
    """The per-image assignment kept for the visualizer still runs with the device-grid helpers."""
    new, _ = heads(5, False, 0)
    xin = [x.detach() for x in feats(2, 256, 0)]
    new.visualize_assign_result(xin, make_labels(2, 30, 5, 256, [4, 2], 0), torch.rand(2, 3, 256, 256) * 255,
                                save_prefix=str(tmp_path / "v_"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v_0.png", "v_1.png"]
