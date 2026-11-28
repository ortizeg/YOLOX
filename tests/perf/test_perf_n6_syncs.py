"""N6: no host sync in a training step.

The head/loss side (formerly flags iou_nosync and dense_loss) is now the only path; its equivalence with upstream is in
test_head_reference.py. Here:
- the head returns num_fg / num_gts as a device tensor (nothing read back for the loss)
- ``log_onesync`` (tier A): the logging step reads every loss meter with one copy; the values are unchanged
- CUDA (skipped elsewhere): a full head training step raises nothing under ``set_sync_debug_mode("error")`` when the
  trainer provides host label counts; without them (one ``nlabel.tolist()``) and with the upstream head it does,
  which shows the check is meaningful.
"""

import pytest
import torch

from yolox.core.trainer import latest_on_host
from yolox.utils.metric import MeterBuffer

from .reference.upstream_head import UpstreamYOLOXHead
from .test_perf_head_nosync import feats, make_head
from .test_perf_model import make_labels


def test_head_returns_device_num_fg():
    head = make_head(5, False, 0)
    xin = feats(2, 256, 0)
    labels = make_labels(2, 30, 5, 256, [4, 2], 0)
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


def _cuda_step(head, labels, counts, mode):
    xin = feats(len(counts) if counts else labels.shape[0], 320, 0, torch.device("cuda"), torch.float16)
    if counts is not None:
        head.host_label_counts = list(counts)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(mode)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            out = head(xin, labels=labels, imgs=None)
        out[0].backward()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("use_l1", [False, True])
def test_head_step_no_host_sync_cuda(use_l1):
    C, counts = 5, [7, 0, 23, 1]
    head = make_head(C, use_l1, 0).cuda()
    labels = make_labels(len(counts), 60, C, 320, counts, 0).cuda()
    _cuda_step(head, labels, counts, 0)  # warm the grid caches
    out = _cuda_step(head, labels, counts, "error")  # forward + backward, nothing synchronises
    assert torch.isfinite(out[0]).item()
    with pytest.raises(RuntimeError):  # no host counts: SimOTA reads them back (nlabel.tolist())
        _cuda_step(head, labels, None, "error")
    ref = UpstreamYOLOXHead(C, width=0.25, in_channels=[256, 512, 1024]).cuda().train()
    ref.use_l1 = use_l1
    with pytest.raises(RuntimeError):  # the upstream head syncs (per-image loop, .item(), nonzero, prod backward)
        _cuda_step(ref, labels, None, "error")
