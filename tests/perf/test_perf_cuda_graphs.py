"""N7 perf flag cuda_graphs: backbone + neck + head convs replayed as CUDA graphs (CUDA only).

- training steps with the graphed trunk match the eager model (losses, gradients, updated weights), over two input
  sizes, with and without the L1 loss, fp32 and fp16 autocast + channels_last
- capture leaves the model state unchanged (BatchNorm statistics restored)
- an uncovered shape, eval mode and an EMA deepcopy all take the eager path; the head's raw path equals its eager one
"""

import copy

import pytest
import torch

from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import ModelEMA

from .conftest import flags
from .test_perf_model import make_labels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

N6 = "simota_batch,simota_nosync,head_nosync,host_label_counts,iou_nosync,dense_loss"
SIZES = [(256, 256), (320, 320)]
B = 2


def make_model(use_l1, seed=0):
    torch.manual_seed(seed)
    m = YOLOX(YOLOPAFPN(0.33, 0.25), YOLOXHead(5, 0.25)).cuda().train()
    m.head.use_l1 = use_l1
    return m


def batches(n, dtype):
    out = []
    for i in range(n):
        h, w = SIZES[i % len(SIZES)]
        g = torch.Generator().manual_seed(100 + i)
        x = torch.randn(B, 3, h, w, generator=g).cuda().to(dtype)
        counts = [5 + i, 2]
        out.append((x, make_labels(B, 30, 5, h, counts, i).cuda().to(dtype), counts))
    return out


def train(model, spec, data, amp, channels_last, capture):
    from yolox.utils import cuda_graphs
    if channels_last:
        model.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    losses, grads = [], []
    with flags(spec):
        if capture:
            cuda_graphs.capture(model, SIZES, B, data[0][0].dtype, channels_last, torch.float16 if amp else None)
        for x, labels, counts in data:
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            model.head.host_label_counts = list(counts)
            assert (cuda_graphs.lookup(model, x) is not None) == capture  # the graph path really runs
            with torch.autocast("cuda", enabled=amp, dtype=torch.float16, cache_enabled="cuda_graphs" not in spec):
                out = model(x, labels)
            opt.zero_grad(set_to_none=True)
            out["total_loss"].backward()
            grads.append([p.grad.detach().clone() for p in model.parameters()])
            opt.step()
            losses.append(out["total_loss"].detach().float())
    cuda_graphs.release(model)
    return losses, grads, {k: v.detach().clone() for k, v in model.state_dict().items()}


@pytest.mark.parametrize("use_l1,amp,channels_last", [(False, False, False), (True, False, False),
                                                      (False, True, True), (True, True, True)])
def test_graphed_training_matches_eager(use_l1, amp, channels_last):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    data = batches(4, torch.float16 if amp else torch.float32)
    ref = train(make_model(use_l1), N6, data, amp, channels_last, capture=False)
    new = train(make_model(use_l1), N6 + ",cuda_graphs", data, amp, channels_last, capture=True)
    tol = dict(rtol=2e-3, atol=2e-4) if amp else dict(rtol=1e-4, atol=1e-6)
    for a, b in zip(ref[0], new[0]):
        torch.testing.assert_close(a, b, **tol)
    for ga, gb in zip(ref[1], new[1]):
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, **tol)
    for k in ref[2]:
        torch.testing.assert_close(ref[2][k].float(), new[2][k].float(), **tol)


def test_capture_leaves_state_unchanged_and_fallbacks():
    from yolox.utils import cuda_graphs
    model = make_model(False)
    with torch.no_grad():  # some non-trivial running statistics
        model(torch.randn(B, 3, 256, 256, device="cuda"), make_labels(B, 30, 5, 256, [3, 1], 0).cuda())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with flags(N6 + ",cuda_graphs"):
        cuda_graphs.capture(model, SIZES, B, torch.float32, False, None)
        for k, v in model.state_dict().items():
            assert torch.equal(before[k], v), k
        x = torch.randn(B, 3, 256, 256, device="cuda")
        assert cuda_graphs.lookup(model, x) is not None
        assert cuda_graphs.lookup(model, torch.randn(B, 3, 384, 384, device="cuda")) is None  # other size
        assert cuda_graphs.lookup(model, torch.randn(B + 1, 3, 256, 256, device="cuda")) is None  # other batch
        ema = ModelEMA(model, 0.9998)
        assert cuda_graphs.lookup(ema.ema, x) is None  # the deepcopy is a plain model
        copy.deepcopy(model)  # the registry is not part of the model
        model.eval()
        assert cuda_graphs.lookup(model, x) is None  # eval: eager decode
        model.train()
    cuda_graphs.release(model)
    assert cuda_graphs.lookup(model, x) is None


def test_head_raw_path_equals_eager():
    model = make_model(True)
    x = torch.randn(B, 3, 256, 256, device="cuda")
    labels = make_labels(B, 30, 5, 256, [4, 2], 0).cuda()
    with flags(N6):
        model.head.host_label_counts = [4, 2]
        fpn = model.backbone(x)
        ref = model.head(fpn, labels, x)
        raw = model.head.raw_outputs(fpn)
        model.head.host_label_counts = [4, 2]
        new = model.head(raw, labels, x, raw=raw)
    for a, b in zip(ref, new):
        assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))


def test_train_sizes():
    from yolox.utils.cuda_graphs import train_sizes

    class E:
        input_size = (640, 640)
        multiscale_range = 5
    assert train_sizes(E()) == [(s, s) for s in range(480, 801, 32)]
    E.multiscale_range = 0
    assert train_sizes(E()) == [(640, 640)]
