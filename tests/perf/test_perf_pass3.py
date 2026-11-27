"""Pass-3 perf flags, CPU equivalence.

Tier A (bit-identical): cv_flip, preproc_nopad, hwc_transport (+ the trainer's to_nchw), int_indices, fast_anno,
resize_nosync. simota_nosync is tier B like simota_batch (CUDA topk tie order) but bit-identical on CPU.
"""

import json
import os
import random
import socket

import numpy as np
import pytest
import torch

from yolox.core.trainer import to_nchw
from yolox.data import COCODataset
from yolox.data.data_augment import _mirror, preproc
from yolox.data.samplers import InfiniteSampler, YoloBatchSampler
from yolox.exp import Exp

from .conftest import flags, seed_all
from .test_perf_data import DATA_FLAGS, build, samples
from .test_perf_model import CASES, head_losses

PASS3_DATA = "cv_flip,preproc_nopad,hwc_transport"


def as_chw_float(img):
    """Loader output -> the original CHW float32 layout (HWC uint8 from hwc_transport is transposed back)."""
    if img.dtype == np.uint8 and img.ndim == 3 and img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)
    return np.ascontiguousarray(img, dtype=np.float32)


# ---------------------------------------------------------------- data path (tier A)

@pytest.mark.parametrize("size,mosaic", [((640, 640), True), ((800, 800), True), ((640, 640), False)])
def test_pass3_pipeline_identical(coco_dir, size, mosaic):
    """Original pipeline vs validated data flags + all pass-3 data flags: identical pixels and labels."""
    with flags(""):
        ref = samples(build(coco_dir, size, mosaic), 48, 2000)
    with flags(DATA_FLAGS + "," + PASS3_DATA):
        new = samples(build(coco_dir, size, mosaic), 48, 2000)
    for (ri, rl, rinfo), (ni, nl, ninfo) in zip(ref, new):
        assert ri.dtype == np.float32 and ni.dtype == np.uint8 and ni.shape == (size[0], size[1], 3)
        assert ni.flags.c_contiguous
        assert np.array_equal(ri, as_chw_float(ni))
        assert np.array_equal(rl, nl)
        assert rinfo == ninfo


@pytest.mark.parametrize("flag", PASS3_DATA.split(",") + ["uint8_transport,hwc_transport"])
def test_each_pass3_data_flag_alone(coco_dir, flag):
    with flags(""):
        ref = samples(build(coco_dir), 24, 9)
    with flags(flag):
        new = samples(build(coco_dir), 24, 9)
    for (ri, rl, _), (ni, nl, _) in zip(ref, new):
        assert np.array_equal(ri, as_chw_float(ni)), flag
        assert np.array_equal(rl, nl), flag


def test_hwc_needs_uint8_transport():
    """hwc_transport alone leaves the float32 CHW output (and every eval/ValTransform call) unchanged."""
    img = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    with flags(""):
        ref, _ = preproc(img, (640, 640))
    with flags("hwc_transport,preproc_nopad"):
        new, _ = preproc(img, (640, 640))
    assert new.dtype == np.float32 and np.array_equal(ref, new)


def test_cv_flip_mirror_identical():
    rng = np.random.default_rng(3)
    for s in range(20):
        img = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
        boxes = rng.uniform(0, 90, (7, 4)).astype(np.float32)
        with flags(""):
            random.seed(s)
            a, ba = _mirror(img, boxes.copy(), 0.5)
        with flags("cv_flip"):
            random.seed(s)
            b, bb = _mirror(img, boxes.copy(), 0.5)
        assert np.array_equal(a, b) and np.array_equal(ba, bb)


@pytest.mark.parametrize("shape", [(640, 640, 3), (800, 800, 3), (640, 480, 3), (360, 640, 3)])
def test_preproc_nopad_identical(shape):
    img = np.random.default_rng(4).integers(0, 256, shape, dtype=np.uint8)
    flipped = img[:, ::-1]  # non-contiguous input, as _mirror returns without cv_flip
    size = (shape[0], shape[0]) if shape[0] == shape[1] else (640, 640)
    for x in (img, flipped):
        with flags(""):
            ref, r0 = preproc(x, size)
        for f in ("preproc_nopad", "preproc_nopad,uint8_transport", "preproc_nopad,hwc_transport"):
            with flags(f):
                new, r1 = preproc(x, size, as_uint8="uint8" in f or "hwc" in f)
            assert r0 == r1 and np.array_equal(ref, as_chw_float(new)), f


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_to_nchw_identical(dtype):
    g = torch.Generator().manual_seed(0)
    hwc = torch.randint(0, 256, (3, 96, 128, 3), generator=g, dtype=torch.uint8)
    chw = hwc.permute(0, 3, 1, 2).contiguous()  # what the CHW loader sends
    with flags("uint8_transport"):
        ref = to_nchw(chw, dtype)
    with flags("uint8_transport,hwc_transport"):
        new = to_nchw(hwc, dtype)
        same = to_nchw(chw, dtype)  # a CHW batch passes through unchanged (e.g. the no-flag float path)
    assert new.shape == ref.shape and new.dtype == dtype and new.is_contiguous()
    assert torch.equal(ref, new) and torch.equal(ref, same)


# ---------------------------------------------------------------- sampler (tier A)

def test_int_indices_same_stream():
    for shuffle in (True, False):
        with flags(""):
            ref = InfiniteSampler(37, shuffle=shuffle, seed=5)
            a = [int(i) for _, i in zip(range(37 * 3 + 4), iter(ref))]
        with flags("int_indices"):
            new = InfiniteSampler(37, shuffle=shuffle, seed=5)
            out = [i for _, i in zip(range(37 * 3 + 4), iter(new))]
        assert a == out and all(type(i) is int for i in out)


def test_int_indices_same_samples(coco_dir):
    """The dataset gets (mosaic, int) instead of (mosaic, 0-d tensor): same samples."""
    ds = build(coco_dir)
    outs = {}
    for spec in ("", "int_indices"):
        with flags(spec):
            bs = YoloBatchSampler(sampler=InfiniteSampler(len(ds), seed=1), batch_size=4, drop_last=False, mosaic=True)
            it = iter(bs)
            got = []
            for b in range(3):
                for j, idx in enumerate(next(it)):
                    seed_all(100 * b + j)
                    img, lab, _, _ = ds[idx]
                    got.append((img, lab))
            outs[spec] = got
    for (ri, rl), (ni, nl) in zip(outs[""], outs["int_indices"]):
        assert np.array_equal(ri, ni) and np.array_equal(rl, nl)


# ---------------------------------------------------------------- annotation parsing (tier A)

def _edge_json(root):
    """Edge cases for the bbox cleaning: int and float boxes, negative origins, boxes past the image border,
    zero / negative sizes, zero area, -0.0, a NaN box, integer image sizes."""
    (root / "annotations").mkdir(parents=True)
    rng = np.random.default_rng(7)
    images, anns, aid = [], [], 1
    special = [[0, 0, 10, 10], [-5, -3, 20, 30], [-5.5, 2.25, 1e3, 1e3], [630, 470, 50, 50], [10, 10, 0, 0],
               [10, 10, -4, 5], [0.0, -0.0, 3.5, 2.0], [float("nan"), 3, 4, 5], [100, 100, 7.25, 1e-9],
               [639.999, 479.5, 0.001, 0.5]]
    for i in range(30):
        w, h = (640, 480) if i % 2 else (1280, 720)
        images.append({"id": i + 1, "file_name": f"{i:04d}.jpg", "height": h, "width": w})
        boxes = special if i == 0 else [
            [float(rng.uniform(-20, w)), float(rng.uniform(-20, h)), float(rng.uniform(-2, w / 2)),
             float(rng.uniform(-2, h / 2))] if rng.random() < 0.7 else
            [int(rng.integers(-5, w)), int(rng.integers(-5, h)), int(rng.integers(0, 300)), int(rng.integers(0, 300))]
            for _ in range(int(rng.integers(0, 12)))]
        for b in boxes:
            area = 0.0 if aid % 11 == 0 else abs(float(np.nan_to_num(b[2] * b[3]))) + 1.0
            anns.append({"id": aid, "image_id": i + 1, "category_id": int(rng.choice([2, 5, 9])), "bbox": b,
                         "area": area, "iscrowd": 0})
            aid += 1
    js = {"images": images, "annotations": anns, "categories": [{"id": c, "name": f"c{c}"} for c in (2, 5, 9)]}
    (root / "annotations" / "train.json").write_text(json.dumps(js))  # NaN is written as the NaN literal
    return root


def _same_annotations(a, b):
    assert len(a.annotations) == len(b.annotations)
    for (ra, ia, za, fa), (rb, ib, zb, fb) in zip(a.annotations, b.annotations):
        assert ra.dtype == rb.dtype and ra.shape == rb.shape
        assert np.array_equal(ra, rb, equal_nan=True)
        assert np.array_equal(np.signbit(ra), np.signbit(rb))
        assert ia == ib and za == zb and fa == fb
    ca = [an.get("clean_bbox") for an in a.coco.dataset["annotations"]]
    cb = [an.get("clean_bbox") for an in b.coco.dataset["annotations"]]
    assert [c is None for c in ca] == [c is None for c in cb]
    for x, y in zip(ca, cb):
        if x is not None:
            assert np.array_equal(np.asarray(x, float), np.asarray(y, float), equal_nan=True)


@pytest.mark.parametrize("extra", ["", ",class_map"])
def test_fast_anno_identical(coco_dir, tmp_path, extra):
    edge = _edge_json(tmp_path / "edge")
    for root in (coco_dir, edge):
        for size in ((640, 640), (800, 800)):
            with flags(extra.strip(",")):
                ref = COCODataset(data_dir=str(root), json_file="train.json", name="train", img_size=size)
            with flags("fast_anno" + extra):
                new = COCODataset(data_dir=str(root), json_file="train.json", name="train", img_size=size)
            _same_annotations(ref, new)


# ---------------------------------------------------------------- batched SimOTA without syncs (B; CPU exact)

@pytest.mark.parametrize("C,counts,use_l1,seed", CASES + [(365, [40, 3, 0, 17], True, 5)])
def test_simota_nosync_identical(C, counts, use_l1, seed):
    ref, rg = head_losses("simota_batch", C, counts, use_l1, seed)
    new, ng = head_losses("simota_batch,simota_nosync", C, counts, use_l1, seed)
    loop, lg = head_losses("", C, counts, use_l1, seed)
    for a, b, c in zip(ref, new, loop):
        assert torch.equal(a, b) and torch.equal(c, b), (a, b, c)
    for a, b, c in zip(rg, ng, lg):
        assert torch.equal(a, b) and torch.equal(c, b)


def test_simota_nosync_crowded_and_empty():
    for counts, crowd in (([30, 8], True), ([0, 0], False), ([1], False)):
        ref, rg = head_losses("simota_batch", 5, counts, True, 11, crowd=crowd)
        new, ng = head_losses("simota_batch,simota_nosync", 5, counts, True, 11, crowd=crowd)
        assert all(torch.equal(a, b) for a, b in zip(ref, new)), counts
        assert all(torch.equal(a, b) for a, b in zip(rg, ng)), counts


def test_simota_nosync_with_cls_cost_algebra():
    ref, rg = head_losses("simota_batch,cls_cost_algebra", 80, [15, 30, 2], True, 2)
    new, ng = head_losses("simota_batch,cls_cost_algebra,simota_nosync", 80, [15, 30, 2], True, 2)
    assert all(torch.equal(a, b) for a, b in zip(ref, new))
    assert all(torch.equal(a, b) for a, b in zip(rg, ng))


# ---------------------------------------------------------------- multiscale size draw (tier A)

def _sizes(exp, spec, n, distributed=False):
    with flags(spec):
        random.seed(123)
        exp.__dict__.pop("random_size", None)
        return [exp.random_resize(None, 0, 0, distributed) for _ in range(n)]


def test_resize_nosync_same_sizes(monkeypatch):
    """The original draws into a CUDA LongTensor; run it on CPU (Tensor.cuda -> identity) and compare."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    for inp, rng in (((640, 640), 5), ((800, 800), 5), ((640, 960), 3)):
        exp = Exp()
        exp.input_size, exp.multiscale_range = inp, rng
        ref = _sizes(exp, "", 60)
        new = _sizes(exp, "resize_nosync", 60)
        assert ref == new and all(type(v) is int for s in new for v in s)
        assert len(set(new)) > 3


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_resize_nosync_distributed_gloo(monkeypatch):
    """World size 1 gloo group on loopback: the gloo-broadcast branch returns the same sizes as the original
    barrier + broadcast branch."""
    import torch.distributed as dist
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_port()))
    if os.uname().sysname == "Darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    try:
        dist.init_process_group("gloo", rank=0, world_size=1)
    except Exception as e:  # pragma: no cover - environment without loopback gloo
        pytest.skip(f"gloo unavailable: {e}")
    try:
        exp = Exp()
        exp.input_size = (800, 800)
        ref = _sizes(exp, "", 30, distributed=True)
        new = _sizes(exp, "resize_nosync", 30, distributed=True)
        assert ref == new
    finally:
        dist.destroy_process_group()
