"""Tests for yolox/data/dataloading.py and yolox/data/samplers.py."""

import itertools

import pytest
import torch

from yolox.data.dataloading import list_collate
from yolox.data.samplers import InfiniteSampler, YoloBatchSampler


class TestInfiniteSampler:
    """Tests for InfiniteSampler."""

    def test_indices_in_valid_range(self):
        """All produced indices are in [0, size)."""
        sampler = InfiniteSampler(size=10, shuffle=True, seed=0, rank=0, world_size=1)
        indices = list(itertools.islice(sampler, 50))

        for idx in indices:
            assert 0 <= idx < 10, f"Index {idx} out of range [0, 10)"

    def test_wraps_around(self):
        """Produces more than `size` indices (wraps around)."""
        size = 5
        sampler = InfiniteSampler(
            size=size, shuffle=False, seed=0, rank=0, world_size=1
        )
        indices = list(itertools.islice(sampler, size * 3))

        assert len(indices) == size * 3

    def test_shuffle_false_produces_sequential(self):
        """With shuffle=False, produces sequential indices."""
        size = 8
        sampler = InfiniteSampler(
            size=size, shuffle=False, seed=0, rank=0, world_size=1
        )
        indices = list(itertools.islice(sampler, size * 2))

        # Should cycle through 0..size-1 repeatedly
        expected = list(range(size)) * 2
        assert indices == expected

    def test_different_ranks_get_different_indices(self):
        """Different ranks produce different index streams."""
        size = 20
        sampler_r0 = InfiniteSampler(
            size=size, shuffle=True, seed=0, rank=0, world_size=2
        )
        sampler_r1 = InfiniteSampler(
            size=size, shuffle=True, seed=0, rank=1, world_size=2
        )

        indices_r0 = list(itertools.islice(sampler_r0, 30))
        indices_r1 = list(itertools.islice(sampler_r1, 30))

        # Different ranks should yield different streams
        assert indices_r0 != indices_r1


class TestYoloBatchSampler:
    """Tests for YoloBatchSampler."""

    def test_yields_batches_of_tuples(self):
        """Yields batches of (mosaic_flag, index) tuples."""
        sampler = InfiniteSampler(
            size=100, shuffle=False, seed=0, rank=0, world_size=1
        )
        batch_sampler = YoloBatchSampler(
            sampler=iter(sampler), batch_size=4, drop_last=True, mosaic=True
        )

        batch = next(iter(batch_sampler))

        assert len(batch) == 4
        for item in batch:
            assert isinstance(item, (list, tuple))
            assert len(item) == 2  # (mosaic_flag, index)

    def test_mosaic_flag_matches_setting(self):
        """Mosaic flag in batch items matches the sampler's mosaic setting."""
        sampler = InfiniteSampler(
            size=100, shuffle=False, seed=0, rank=0, world_size=1
        )

        # Test with mosaic=True
        batch_sampler_on = YoloBatchSampler(
            sampler=iter(sampler), batch_size=4, drop_last=True, mosaic=True
        )
        batch = next(iter(batch_sampler_on))
        for flag, idx in batch:
            assert flag is True, "Mosaic flag should be True"

        # Test with mosaic=False
        sampler2 = InfiniteSampler(
            size=100, shuffle=False, seed=0, rank=0, world_size=1
        )
        batch_sampler_off = YoloBatchSampler(
            sampler=iter(sampler2), batch_size=4, drop_last=True, mosaic=False
        )
        batch = next(iter(batch_sampler_off))
        for flag, idx in batch:
            assert flag is False, "Mosaic flag should be False"


class TestListCollate:
    """Tests for list_collate."""

    def test_collates_tensors_and_lists(self):
        """Collates a batch of (tensor, list) pairs correctly."""
        batch = [
            (torch.tensor([1.0, 2.0]), [10, 20]),
            (torch.tensor([3.0, 4.0]), [30, 40]),
        ]

        result = list_collate(batch)

        # list_collate should produce lists of the elements
        assert isinstance(result, (list, tuple))
        assert len(result) == 2  # Two fields in each sample
