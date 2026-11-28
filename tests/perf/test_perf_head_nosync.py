"""Head grids and stride rows built and cached on the device (formerly flag head_nosync, now the only path).

Cache keys and dtypes; on CUDA (skipped elsewhere) the grid, decode and inf-scalar paths run without a host sync
under torch.cuda.set_sync_debug_mode("error"). Equivalence with upstream's grids: test_head_reference.py.
"""

import pytest
import torch

from yolox.models import YOLOXHead


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
    """The head's grid/stride construction, decode and the inf scalar raise no sync under
    set_sync_debug_mode("error"); upstream's constructions do (shows the check is meaningful)."""
    dev = torch.device("cuda")
    head = make_head(5, False, 0).to(dev)
    xin = feats(2, 320, 0, dev, torch.float16)
    out = torch.randn(2, 10, 40, 40, device=dev, dtype=torch.float16)
    cost = torch.randn(2, 3, 100, device=dev)
    torch.cuda.synchronize()

    # warm the caches (first build allocates; still no host sync, but keep the check to the steady state)
    head.get_output_and_grid(out.clone(), 0, 8, xin[0].dtype, xin[0].device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        head.get_output_and_grid(out.clone(), 0, 8, xin[0].dtype, xin[0].device)
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
