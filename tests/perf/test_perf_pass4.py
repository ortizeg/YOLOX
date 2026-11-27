"""decode_reuse (tier C): partners come from a per-worker pool; the primary image is always a fresh decode."""

import numpy as np
import pytest

from yolox.data.datasets.mosaicdetection import MosaicDetection
from yolox.utils import perf

from .conftest import flags, seed_all


class CountingDataset:
    """pull_item counts decodes; image i is filled with value i so samples can be traced."""

    def __init__(self, n=40, size=64, empty=()):
        self.n, self.size, self.empty = n, size, set(empty)
        self.input_dim = (size, size)
        self._input_dim = self.input_dim
        self.pulls = []

    def __len__(self):
        return self.n

    def load_anno(self, i):
        return np.zeros((0, 5), np.float32) if i in self.empty else np.array([[4, 4, 30, 30, 0]], np.float32)

    def pull_item(self, i):
        self.pulls.append(i)
        img = np.full((self.size, self.size, 3), i, np.uint8)
        return img, self.load_anno(i).copy(), (self.size, self.size), np.array([i])


def ident(img, labels, dim):
    return img, labels


def make(ds, **kw):
    m = MosaicDetection(ds, img_size=ds.input_dim, preproc=ident, degrees=0.0, translate=0.0, shear=0.0,
                        mosaic_scale=(1, 1), **kw)
    m._input_dim = ds.input_dim
    return m


def run(m, idxs):
    for i in idxs:
        m.__getitem__.__wrapped__(m, i)  # skip the mosaic_getitem tuple unwrapping


def test_tier_c_not_in_all():
    assert perf.TIER["decode_reuse"] == "C"
    assert "decode_reuse" not in perf.parse("all")
    assert "decode_reuse" in perf.parse("all,decode_reuse")


def test_flag_off_is_original_draws():
    """Off: same pulls and same images as before the change (no extra random calls)."""
    outs = []
    for _ in range(2):
        ds = CountingDataset()
        m = make(ds, reuse_pool=8)
        seed_all(0)
        with flags(""):
            outs.append((list(map(int, [m.__getitem__.__wrapped__(m, i)[0].sum() for i in range(10)])), ds.pulls))
    assert outs[0] == outs[1]
    assert len(outs[0][1]) == 10 * 5  # 4 mosaic + 1 mixup decodes per sample


@pytest.mark.parametrize("prob,expect_max", [(1.0, 1.2), (0.5, 3.2)])
def test_decodes_per_sample(prob, expect_max):
    ds = CountingDataset(n=200)
    m = make(ds, reuse_pool=16, reuse_prob=prob)
    seed_all(1)
    with flags("decode_reuse"):
        run(m, range(200))
    per = len(ds.pulls) / 200
    assert per <= expect_max, per
    assert per >= 1.0


def test_primary_always_fresh_and_never_own_partner():
    ds = CountingDataset(n=50)
    m = make(ds, reuse_pool=4, enable_mixup=False)
    seed_all(2)
    with flags("decode_reuse"):
        for i in range(50):
            before = len(ds.pulls)
            m.__getitem__.__wrapped__(m, i)
            assert ds.pulls[before] == i  # first pull of the sample is the primary
            assert m._pool[(m._pool_next - 1) % len(m._pool)][3][0] == i  # pushed last
    assert len(m._pool) == 4


def test_mixup_partner_has_labels():
    ds = CountingDataset(n=30, empty=set(range(0, 30, 2)))
    m = make(ds, reuse_pool=8)
    seed_all(3)
    with flags("decode_reuse"):
        for _ in range(200):
            _, labels, _, _ = m._pull_partner(0, need_labels=True) if m._pool else (None, np.ones((1, 5)), 0, 0)
            assert labels.size
            run(m, [1])


def test_pool_images_not_mutated():
    ds = CountingDataset(n=30)
    m = make(ds, reuse_pool=8)
    seed_all(4)
    with flags("decode_reuse"):
        run(m, range(60))
        for img, _, _, iid in m._pool:
            assert (img == iid[0]).all()


def test_pool_zero_disables():
    ds = CountingDataset(n=30)
    m = make(ds, reuse_pool=0)
    seed_all(5)
    with flags("decode_reuse"):
        run(m, range(10))
    assert len(ds.pulls) == 50 and not m._pool
