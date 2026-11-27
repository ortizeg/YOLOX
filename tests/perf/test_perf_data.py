"""Tier A: the data-path perf flags must reproduce the original augmentation pipeline bit for bit."""

import numpy as np
import pytest

from yolox.data import COCODataset, MosaicDetection, TrainTransform
from yolox.data.data_augment import augment_hsv, preproc

from .conftest import flags, seed_all

DATA_FLAGS = "uint8_transport,skip_noop_resize,hsv_lut,int_mixup,class_map"


def build(coco_dir, size=(640, 640), mosaic=True):
    ds = COCODataset(data_dir=str(coco_dir), json_file="train.json", name="train", img_size=size,
                     preproc=TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0))
    if not mosaic:
        return ds
    return MosaicDetection(ds, mosaic=True, img_size=size,
                           preproc=TrainTransform(max_labels=120, flip_prob=0.5, hsv_prob=1.0),
                           degrees=10.0, translate=0.2, mosaic_scale=(0.5, 1.5), mixup_scale=(0.5, 1.5),
                           shear=2.0, enable_mixup=True, mosaic_prob=1.0, mixup_prob=0.5)


def samples(ds, n, seed0):
    out = []
    for i in range(n):
        seed_all(seed0 + i)
        img, lab, info, iid = ds[i % len(ds)]
        out.append((img, lab, info))
    return out


@pytest.mark.parametrize("size,mosaic", [((640, 640), True), ((800, 800), True), ((640, 640), False)])
def test_pipeline_identical(coco_dir, size, mosaic):
    with flags(""):
        ref = samples(build(coco_dir, size, mosaic), 48, 1000)
    with flags(DATA_FLAGS):
        new = samples(build(coco_dir, size, mosaic), 48, 1000)
    for (ri, rl, rinfo), (ni, nl, ninfo) in zip(ref, new):
        assert ri.dtype == np.float32 and ni.dtype == np.uint8
        assert np.array_equal(ri, ni.astype(np.float32))
        assert np.array_equal(rl, nl)
        assert rinfo == ninfo


def test_each_data_flag_alone(coco_dir):
    with flags(""):
        ref = samples(build(coco_dir), 24, 7)
    for f in DATA_FLAGS.split(","):
        with flags(f):
            new = samples(build(coco_dir), 24, 7)
        for (ri, rl, _), (ni, nl, _) in zip(ref, new):
            assert np.array_equal(ri, ni.astype(np.float32)), f
            assert np.array_equal(rl, nl), f


def test_hsv_lut_exact():
    rng = np.random.default_rng(1)
    for s in range(50):
        img = rng.integers(0, 256, (97, 131, 3), dtype=np.uint8)
        a, b = img.copy(), img.copy()
        with flags(""):
            np.random.seed(s)
            augment_hsv(a)
        with flags("hsv_lut"):
            np.random.seed(s)
            augment_hsv(b)
        assert np.array_equal(a, b)


def test_preproc_uint8_and_noop():
    rng = np.random.default_rng(2)
    for shape in [(640, 640, 3), (360, 640, 3), (640, 400, 3), (500, 700, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        with flags(""):
            ref, r0 = preproc(img, (640, 640))
        with flags("skip_noop_resize"):
            new, r1 = preproc(img, (640, 640), as_uint8=True)
        assert r0 == r1 and np.array_equal(ref, new.astype(np.float32))


def test_fallback_path_still_identical(coco_dir):
    """Image 5 has only ~1 px boxes: TrainTransform must take the original-copy fallback with the flag on."""
    with flags(""):
        ref = samples(build(coco_dir, mosaic=False), 1, 0)
    ds = build(coco_dir, mosaic=False)
    with flags(DATA_FLAGS):
        out = []
        for s in range(5):
            seed_all(s)
            out.append(ds[5])
    with flags(""):
        ds0 = build(coco_dir, mosaic=False)
        for s in range(5):
            seed_all(s)
            ri, rl, _, _ = ds0[5]
            assert np.array_equal(ri, out[s][0].astype(np.float32)) and np.array_equal(rl, out[s][1])
    assert ref


def test_lazy_disk_cache_identical(coco_dir, tmp_path):
    """--cache lazy: first read decodes and stores, second read loads the stored copy; both equal the uncached
    image, the cache directory is keyed by size, and no temp files are left behind."""
    import shutil
    from yolox.data import COCODataset
    root = tmp_path / "c"
    shutil.copytree(coco_dir, root)
    plain = COCODataset(data_dir=str(root), json_file="train.json", name="train", img_size=(640, 640))
    lazy = COCODataset(data_dir=str(root), json_file="train.json", name="train", img_size=(640, 640),
                       cache=True, cache_type="lazy")
    assert lazy.cache_dir.endswith("cache_train_640x640")
    for _ in range(2):  # miss, then hit
        for i in range(len(plain)):
            assert np.array_equal(plain.read_img(i), lazy.read_img(i))
    files = list((root / "cache_train_640x640").rglob("*"))
    assert len([f for f in files if f.suffix == ".npy"]) == len(plain)
    assert not [f for f in files if f.name.endswith(".tmp")]
    with flags(""):
        ref = samples(build(root), 12, 3)
    ds = COCODataset(data_dir=str(root), json_file="train.json", name="train", img_size=(640, 640),
                     preproc=TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0), cache=True, cache_type="lazy")
    mos = MosaicDetection(ds, mosaic=True, img_size=(640, 640),
                          preproc=TrainTransform(max_labels=120, flip_prob=0.5, hsv_prob=1.0), degrees=10.0,
                          translate=0.2, mosaic_scale=(0.5, 1.5), mixup_scale=(0.5, 1.5), shear=2.0,
                          enable_mixup=True, mosaic_prob=1.0, mixup_prob=0.5)
    with flags(""):
        new = samples(mos, 12, 3)
    for (ri, rl, _), (ni, nl, _) in zip(ref, new):
        assert np.array_equal(ri, ni) and np.array_equal(rl, nl)


def test_anno_cache_identical(coco_dir, tmp_path):
    """anno_cache: the first build writes memory-mapped arrays; a second dataset reads them without parsing the
    JSON (no COCO object until something asks for .coco). Every annotation, id and class field is identical,
    and the full mosaic pipeline gives the same samples."""
    import shutil
    from yolox.data import COCODataset
    root = tmp_path / "c"
    shutil.copytree(coco_dir, root)
    with flags(""):
        ref = COCODataset(data_dir=str(root), json_file="train.json", name="train", img_size=(640, 640))
    with flags("anno_cache"):
        first = COCODataset(data_dir=str(root), json_file="train.json", name="train", img_size=(640, 640))
        hit = COCODataset(data_dir=str(root), json_file="train.json", name="train", img_size=(640, 640))
    assert hit._coco is None  # served from the cache
    for ds in (first, hit):
        assert ds.ids == ref.ids and ds.class_ids == ref.class_ids and ds._classes == ref._classes
        assert len(ds.annotations) == len(ref.annotations)
        for a, b in zip(ref.annotations, ds.annotations):
            assert np.array_equal(a[0], b[0]) and a[0].dtype == b[0].dtype
            assert a[1] == b[1] and a[2] == b[2] and a[3] == b[3]
        for i in range(len(ref)):
            ri, rl, rinfo, rid = ref.pull_item(i)
            ni, nl, ninfo, nid = ds.pull_item(i)
            assert np.array_equal(ri, ni) and np.array_equal(rl, nl) and rinfo == ninfo and np.array_equal(rid, nid)
    assert hit.coco.getImgIds() == ref.ids  # lazy COCO for the evaluator
    assert hit._coco is not None
    with flags(""):
        a = samples(build(root), 12, 5)
    with flags("anno_cache"):
        b = samples(build(root), 12, 5)
    for (ri, rl, _), (ni, nl, _) in zip(a, b):
        assert np.array_equal(ri, ni) and np.array_equal(rl, nl)
    other = COCODataset  # a different img_size must not reuse the 640 cache
    with flags("anno_cache"):
        d8 = other(data_dir=str(root), json_file="train.json", name="train", img_size=(800, 800))
    with flags(""):
        r8 = other(data_dir=str(root), json_file="train.json", name="train", img_size=(800, 800))
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(r8.annotations, d8.annotations))
