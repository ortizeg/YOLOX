"""N6 perf flags: no host sync in a training step.

- ``iou_nosync`` (tier B): IOUloss areas as w*h. Forward bit-identical; gradients equal up to rounding (prod's backward
  divides the product by each factor, w*h's multiplies by the other factor).
- ``dense_loss`` (tier B): SimOTA targets for every anchor plus the positive mask, masked dense losses. Losses and
  gradients equal up to summation order, against both batched SimOTA paths, with and without the L1 loss, with images
  that have no labels and a batch with no labels at all.
- ``log_onesync`` (tier A): the logging step reads every loss meter with one copy; the values are unchanged.
- CUDA (skipped elsewhere): a full head training step with the N6 flags raises nothing under
  ``set_sync_debug_mode("error")``; without them it does (shows the check is meaningful).
"""

import pytest
import torch

from yolox.core.trainer import latest_on_host
from yolox.models.losses import IOUloss
from yolox.utils.metric import MeterBuffer

from .conftest import flags
from .test_perf_head_nosync import feats, make_head
from .test_perf_model import make_labels

BATCHED = "simota_batch,host_label_counts"


def boxes(n, seed):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * 300
    wh = torch.rand(n, 2, generator=g) * 80 + 1
    return torch.cat([xy, wh], 1)


@pytest.mark.parametrize("loss_type", ["iou", "giou"])
def test_iou_nosync(loss_type):
    pred0, tgt = boxes(500, 0), boxes(500, 1)
    pred0[:50] = tgt[:50] + 0.5  # heavy overlap
    pred0[50:60, 2:] = tgt[50:60, 2:]  # a few identical sizes
    res = []
    for spec in ["", "iou_nosync"]:
        pred = pred0.clone().requires_grad_()
        with flags(spec):
            loss = IOUloss(reduction="none", loss_type=loss_type)(pred, tgt)
        loss.sum().backward()
        res.append((loss.detach(), pred.grad))
    assert torch.equal(res[0][0], res[1][0])
    torch.testing.assert_close(res[0][1], res[1][1], rtol=1e-5, atol=1e-7)


def head_step(spec, C, counts, use_l1, seed, size=320):
    head = make_head(C, use_l1, seed)
    xin = feats(len(counts), size, seed)
    labels = make_labels(len(counts), 60, C, size, counts, seed)
    with flags(spec):
        if "host_label_counts" in spec:
            head.host_label_counts = list(counts)
        out = head(xin, labels=labels, imgs=None)
    out[0].backward()
    outs = [torch.as_tensor(o).detach().float().reshape(()) for o in out]
    grads = [x.grad.clone() for x in xin] + [p.grad.clone() for p in head.parameters()]
    return outs, grads


@pytest.mark.parametrize("base", [BATCHED, BATCHED + ",simota_nosync"])
@pytest.mark.parametrize("C,counts,use_l1,seed", [(5, [7, 0, 23, 1], False, 0), (5, [40, 12], True, 1),
                                                  (80, [15, 30, 2], False, 2), (5, [3, 0], True, 3),
                                                  (5, [0, 0], True, 4), (5, [0, 0, 0], False, 5)])
def test_dense_loss_matches(base, C, counts, use_l1, seed):
    ro, rg = head_step(base, C, counts, use_l1, seed)
    no, ng = head_step(base + ",dense_loss,iou_nosync", C, counts, use_l1, seed)
    for a, b in zip(ro, no):  # total, iou, obj, cls, l1, num_fg / num_gts
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-6)
    for a, b in zip(rg, ng):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_dense_loss_returns_device_num_fg():
    head = make_head(5, False, 0)
    xin = feats(2, 256, 0)
    labels = make_labels(2, 30, 5, 256, [4, 2], 0)
    with flags(BATCHED + ",dense_loss"):
        head.host_label_counts = [4, 2]
        out = head(xin, labels=labels, imgs=None)
    assert torch.is_tensor(out[5]) and torch.isfinite(out[0])


def test_latest_on_host():
    m = MeterBuffer(window_size=5)
    m.update(total_loss=torch.tensor(3.25), iou_loss=torch.tensor([1.5]), l1_loss=0.0, conf_loss=torch.tensor(0.75))
    m.update(total_loss=torch.tensor(2.5), iou_loss=torch.tensor([1.25]), l1_loss=0.0, conf_loss=torch.tensor(0.5))
    meters = m.get_filtered_meter("loss")
    host = latest_on_host(meters)
    assert list(host) == list(meters)
    for k in meters:
        assert float(host[k].latest) == float(meters[k].latest)
        assert not torch.is_tensor(host[k].latest)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("use_l1", [False, True])
def test_head_step_no_host_sync_cuda(use_l1):
    dev = torch.device("cuda")
    C, counts = 5, [7, 0, 23, 1]
    head = make_head(C, use_l1, 0).to(dev)
    labels = make_labels(len(counts), 60, C, 320, counts, 0).to(dev)

    full = "simota_batch,simota_nosync,head_nosync,host_label_counts,iou_nosync,dense_loss"
    xin = feats(len(counts), 320, 0, dev, torch.float16)
    with flags(full):  # warm the grid caches
        head.host_label_counts = list(counts)
        with torch.autocast("cuda", dtype=torch.float16):
            head(xin, labels=labels, imgs=None)[0].backward()
    torch.cuda.synchronize()
    # forward too: check the whole step under "error"
    xin = feats(len(counts), 320, 0, dev, torch.float16)
    with flags(full):
        head.host_label_counts = list(counts)
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                out = head(xin, labels=labels, imgs=None)
            out[0].backward()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    assert torch.isfinite(out[0]).item()
    # without the N6 flags the same step syncs (nonzero in SimOTA, prod in the IoU loss backward)
    with pytest.raises(RuntimeError):
        xin = feats(len(counts), 320, 0, dev, torch.float16)
        with flags("simota_batch,simota_nosync,head_nosync,host_label_counts"):
            head.host_label_counts = list(counts)
            torch.cuda.set_sync_debug_mode("error")
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = head(xin, labels=labels, imgs=None)
                out[0].backward()
            finally:
                torch.cuda.set_sync_debug_mode(0)
