"""Pass-5 perf flag head_nosync (tier A): head grids and stride rows built and cached on the device.

CPU: training forward + losses + gradients and inference decode are bit-identical with the flag on and off, across
several input sizes (multi-size cache) and with/without simota_batch. CUDA (skipped elsewhere): the changed head
paths run without a host sync under torch.cuda.set_sync_debug_mode("error").
"""

import pytest
import torch

from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

from .conftest import flags
from .test_perf_model import make_labels

SIZES = [320, 256, 384]


def make_head(C, use_l1, seed):
    torch.manual_seed(seed)
    head = YOLOXHead(C, width=0.25, in_channels=[256, 512, 1024])
    head.initialize_biases(1e-2)
    head.use_l1 = use_l1
    return head.train()


def feats(B, size, seed, device="cpu", dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, int(c * 0.25), size // s, size // s, generator=g).to(device, dtype).requires_grad_()
            for c, s in zip([256, 512, 1024], [8, 16, 32])]


def run_sizes(spec, C, counts, use_l1, seed):
    """One head, several steps at different sizes (same head object, so the grid cache is exercised and revisited)."""
    head = make_head(C, use_l1, seed)
    res = []
    with flags(spec):
        for i, size in enumerate(SIZES + SIZES[:1]):
            head.zero_grad()
            xin = feats(len(counts), size, seed + i)
            labels = make_labels(len(counts), 120, C, size, counts, seed + i)
            out = head(xin, labels=labels, imgs=None)
            out[0].backward()
            outs = [o.detach() if torch.is_tensor(o) else torch.tensor(float(o)) for o in out]
            grads = [x.grad.clone() for x in xin] + [p.grad.clone() for p in head.parameters()]
            res.append((outs, grads))
    # flag on: one cache entry per (level, size); flag off: the original path never touches the cache
    assert len(head.__dict__.get("_nosync_cache", {})) == (3 * len(SIZES) if "head_nosync" in spec else 0)
    return res


@pytest.mark.parametrize("base", ["", "simota_batch", "simota_batch,simota_nosync"])
@pytest.mark.parametrize("C,counts,use_l1,seed", [(5, [7, 0, 23, 1], False, 0), (5, [40, 12], True, 1),
                                                  (80, [15, 30, 2], False, 2)])
def test_head_nosync_train_identical(base, C, counts, use_l1, seed):
    ref = run_sizes(base, C, counts, use_l1, seed)
    new = run_sizes(",".join(x for x in [base, "head_nosync"] if x), C, counts, use_l1, seed)
    for (ro, rg), (no, ng) in zip(ref, new):
        for a, b in zip(ro, no):
            assert torch.equal(a, b), (a, b)
        for a, b in zip(rg, ng):
            assert torch.equal(a, b)


def test_head_nosync_decode_identical():
    head = make_head(5, False, 3).eval()
    for size in SIZES + SIZES[:1]:
        xin = feats(2, size, size)
        with torch.no_grad():
            with flags(""):
                ref = head(xin)
            with flags("head_nosync"):
                new = head(xin)
        assert torch.equal(ref, new)


def test_full_model_identical():
    torch.manual_seed(0)
    model = YOLOX(YOLOPAFPN(0.33, 0.25), YOLOXHead(5, 0.25)).train()
    imgs = torch.randn(2, 3, 256, 256)
    labels = make_labels(2, 50, 5, 256, [6, 3], 0)
    with flags("simota_batch"):
        ref = model(imgs, labels)
    with flags("simota_batch,head_nosync"):
        new = model(imgs, labels)
    for k in ref:
        assert torch.equal(torch.as_tensor(ref[k]), torch.as_tensor(new[k])), k


def test_cache_keys_and_dtype():
    head = make_head(5, False, 0)
    g1, r1 = head._device_grid(("train", 0), 4, 6, 8, torch.float16, torch.device("cpu"))
    assert g1.dtype == torch.float16 and r1.dtype == torch.float16
    assert g1.shape == (1, 24, 2) and r1.shape == (1, 24)
    assert torch.equal(r1, torch.full((1, 24), 8.0, dtype=torch.float16))
    # x varies fastest, as in the original stack((xv, yv))
    assert g1[0, :7].tolist() == [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [0, 1]]
    g2, _ = head._device_grid(("train", 0), 4, 6, 8, torch.float16, torch.device("cpu"))
    assert g2 is g1  # cached
    g3, _ = head._device_grid(("train", 0), 4, 6, 8, torch.float32, torch.device("cpu"))
    assert g3 is not g1 and g3.dtype == torch.float32
    g4, _ = head._device_grid(("train", 0), 5, 6, 8, torch.float16, torch.device("cpu"))
    assert g4.shape == (1, 30, 2)
    assert len(head._nosync_cache) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cache_on_cuda_device():
    head = make_head(5, False, 0)
    dev = torch.device("cuda", torch.cuda.current_device())
    g, r = head._device_grid(("train", 0), 4, 6, 8, torch.float16, dev)
    assert g.device == dev and r.device == dev and g.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_head_paths_no_host_sync_cuda():
    """The changed head paths (grid/stride construction, decode, the inf scalar) raise no sync under
    set_sync_debug_mode("error"); the original paths do (shows the check is meaningful). Scoped to these paths only:
    other syncs remain elsewhere in the loss (nlabel.tolist(), nonzero)."""
    dev = torch.device("cuda")
    head = make_head(5, False, 0).to(dev)
    xin = feats(2, 320, 0, dev, torch.float16)
    out = torch.randn(2, 10, 40, 40, device=dev, dtype=torch.float16)
    cost = torch.randn(2, 3, 100, device=dev)
    torch.cuda.synchronize()

    # warm the caches (first build allocates; still no host sync, but keep the check to the steady state)
    with flags("head_nosync"):
        head.get_output_and_grid_nosync(out.clone(), 0, 8, xin[0].dtype, xin[0].device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with flags("head_nosync"):
            head.get_output_and_grid_nosync(out.clone(), 0, 8, xin[0].dtype, xin[0].device)
            head._device_grid(("train", 1), 33, 17, 16, xin[0].dtype, xin[0].device)  # cold build: no sync either
            cost.new_full((), float("inf"))
            head.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                head(xin)  # fp16 inputs + fp32 weights under autocast, as in training; decode on the device
            head.train()
    finally:
        torch.cuda.set_sync_debug_mode("default")

    # the original constructions sync (H2D copies from pageable memory)
    for fn in (lambda: torch.zeros(1, 1600).fill_(8).type_as(xin[0]),
               lambda: torch.tensor(float("inf"), device=dev)):
        torch.cuda.set_sync_debug_mode("error")
        try:
            with pytest.raises(RuntimeError):
                fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
