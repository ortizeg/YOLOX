"""host_label_counts (tier A): per-image label counts computed from the CPU targets equal the head's device rule,
and handing them to the head gives bit-identical losses and gradients. The flag controls whether the trainer
provides the counts; the head uses them whenever they are set and counts on the device otherwise."""

import warnings

import pytest
import torch

from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import perf

from .conftest import flags
from .test_perf_model import make_labels

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def device_rule(labels):
    return (labels.sum(dim=2) > 0).sum(dim=1)


def random_targets(seed, B=6, M=50):
    g = torch.Generator().manual_seed(seed)
    t = torch.zeros(B, M, 5)
    for b in range(B):
        n = int(torch.randint(0, M + 1, (1,), generator=g)) if b else 0  # image 0 empty
        t[b, :n, 0] = torch.randint(0, 5, (n,), generator=g).float()
        t[b, :n, 1:] = torch.rand(n, 4, generator=g) * 600 + 1
    if B > 1:
        t[1, 2] = 0  # an all-zero row between valid rows
    if B > 2:
        t[2, :, 0] = 0  # class 0 everywhere: those rows still count through the box
    if B > 3:
        t[3, -1] = 0  # padding row at the end of a full image
    return t


@pytest.mark.parametrize("seed", range(5))
def test_counts_match_rule(seed):
    t = random_targets(seed)
    host = perf.label_counts(t)
    assert host == device_rule(t).tolist()
    assert isinstance(host, list) and all(type(c) is int for c in host)
    assert host[0] == 0
    # the trainer casts to the training dtype and the multiscale resize scales coordinates; the counts do not move
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for scale in (0.5, 1.0, 1.4):
            u = t.to(dtype)
            u[..., 1::2] = u[..., 1::2] * scale
            u[..., 2::2] = u[..., 2::2] * scale
            assert device_rule(u).tolist() == host


@pytest.mark.parametrize("B", [1, 2, 4, 5, 6])
def test_counts_match_rule_batch_sizes(B):
    """The helper is valid for every batch size the prefetcher test uses (and smaller)."""
    t = random_targets(B, B=B)
    assert t.shape[0] == B
    assert perf.label_counts(t) == device_rule(t).tolist()
    assert perf.label_counts(t.pin_memory() if torch.cuda.is_available() else t) == device_rule(t).tolist()


def test_counts_refuse_negative_entries():
    t = random_targets(0)
    t[1, 0, 1] = -3.0
    assert perf.label_counts(t) is None  # head falls back to the device count


def tiny_yolox(C=5):
    torch.manual_seed(0)
    head = YOLOXHead(C, width=0.25, in_channels=[256, 512, 1024])
    model = YOLOX(YOLOPAFPN(0.33, 0.25, in_channels=[256, 512, 1024]), head)
    head.initialize_biases(1e-2)
    return model


def test_set_label_counts_unwraps():
    model = tiny_yolox()
    perf.set_label_counts(model, [1, 2])
    assert model.head.host_label_counts == [1, 2]
    dp = torch.nn.DataParallel(model)  # CPU-only: a passthrough wrapper with .module, like DDP
    perf.set_label_counts(dp, [3])
    assert model.head.host_label_counts == [3]
    compiled = torch.compile(dp)  # lazy: nothing is compiled until called
    perf.set_label_counts(compiled, [4, 5])
    assert model.head.host_label_counts == [4, 5]
    perf.set_label_counts(compiled, None)
    assert model.head.host_label_counts is None


def head_run(spec, counts_arg, C, counts, use_l1, seed, size=256):
    """Head forward + backward; counts_arg: 'host' (from CPU targets), None (not provided)."""
    torch.manual_seed(seed)
    head = YOLOXHead(C, width=0.25, in_channels=[256, 512, 1024])
    head.initialize_biases(1e-2)
    head.use_l1 = use_l1
    head.train()
    B = len(counts)
    xin = [torch.randn(B, int(c * 0.25), size // s, size // s, requires_grad=True)
           for c, s in zip([256, 512, 1024], [8, 16, 32])]
    labels = make_labels(B, 60, C, size, counts, seed)
    with flags(spec):
        if counts_arg == "host":
            head.host_label_counts = perf.label_counts(labels)
        out = head(xin, labels=labels, imgs=None)
        out[0].backward()
    assert getattr(head, "host_label_counts", None) is None  # consumed
    grads = [x.grad.clone() for x in xin] + [p.grad.clone() for p in head.parameters()]
    return [o if torch.is_tensor(o) else torch.tensor(float(o)) for o in out], grads


def assert_identical(a, b):
    (ra, ga), (rb, gb) = a, b
    for x, y in zip(ra, rb):
        assert torch.equal(x, y), (x, y)
    for x, y in zip(ga, gb):
        assert torch.equal(x, y)


CASES = [(5, [7, 0, 23, 1], False, 0), (5, [40, 12], True, 1), (80, [15, 30, 2], False, 2), (5, [0, 0], False, 3)]


@pytest.mark.parametrize("C,counts,use_l1,seed", CASES)
def test_head_identical(C, counts, use_l1, seed):
    ref = head_run("", None, C, counts, use_l1, seed)
    new = head_run("host_label_counts", "host", C, counts, use_l1, seed)
    assert_identical(ref, new)


def test_full_model_identical():
    """Through YOLOX.forward with the trainer's helper (fp32 CPU, so the step is deterministic)."""
    def run(spec):
        model = tiny_yolox()
        model.train()
        torch.manual_seed(1)
        x = torch.randn(3, 3, 256, 256)
        labels = make_labels(3, 50, 5, 256, [9, 0, 30], 1)
        with flags(spec):
            if perf.on("host_label_counts"):
                perf.set_label_counts(model, perf.label_counts(labels))
            out = model(x, labels)
            out["total_loss"].backward()
        return ([out[k] if torch.is_tensor(out[k]) else torch.tensor(float(out[k])) for k in sorted(out)],
                [p.grad.clone() for p in model.parameters() if p.grad is not None])
    assert_identical(run(""), run("host_label_counts"))


def test_missing_counts_fall_back():
    """No counts set (e.g. a caller driving the model directly): device count, same results; a wrong-length list
    is a caller bug and raises."""
    ref = head_run("", None, 5, [7, 0, 3], False, 5)
    assert_identical(ref, head_run("host_label_counts", None, 5, [7, 0, 3], False, 5))
    head = YOLOXHead(5, width=0.25)
    head.host_label_counts = [1, 2]
    with pytest.raises(ValueError, match="2 host label counts for a batch of 3"):
        head.take_host_label_counts(3)
    assert head.host_label_counts is None


@cuda
def test_prefetcher_counts_cuda():
    from yolox.data import DataPrefetcher
    batches = [(torch.zeros(len(t), 3, 8, 8).pin_memory(), random_targets(i, B=len(t)).pin_memory(), None, None)
               for i, t in enumerate([range(4), range(2), range(5)])]
    with flags("host_label_counts"):
        pf = DataPrefetcher(batches)
        for _, t, _, _ in batches:
            _, target = pf.next()
            torch.cuda.synchronize()
            assert pf.label_counts == device_rule(target).tolist() == device_rule(t).tolist()


def count_sync_warnings(spec, counts):
    head = YOLOXHead(5, width=0.25, in_channels=[256, 512, 1024]).cuda().train()
    B = len(counts)
    xin = [torch.randn(B, int(c * 0.25), 256 // s, 256 // s, device="cuda")
           for c, s in zip([256, 512, 1024], [8, 16, 32])]
    labels_cpu = make_labels(B, 60, 5, 256, counts, 0)
    labels = labels_cpu.cuda()
    with flags(spec):
        if perf.on("host_label_counts"):
            head.host_label_counts = perf.label_counts(labels_cpu)
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                head(xin, labels=labels, imgs=None)
            finally:
                torch.cuda.set_sync_debug_mode("default")
    return sum("synchroniz" in str(w.message) for w in rec)


@cuda
def test_no_count_sync_cuda():
    """With host counts the head's count read-back is gone (the only sync in the head's forward)."""
    counts = [5, 0, 9]
    off = count_sync_warnings("", counts)
    on = count_sync_warnings("host_label_counts", counts)
    assert (off, on) == (1, 0), (off, on)
