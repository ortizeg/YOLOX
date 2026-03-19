"""Tests for yolox/utils/compat.py."""

import pytest
import torch

from yolox.utils.compat import meshgrid


class TestMeshgrid:
    """Tests for meshgrid wrapper."""

    def test_meshgrid_with_1d_tensors(self):
        """meshgrid with 1D tensors returns correct grid."""
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5])

        grid_x, grid_y = meshgrid(x, y)

        assert grid_x.shape == (3, 2)
        assert grid_y.shape == (3, 2)

        # With ij indexing, first tensor varies along rows, second along columns
        expected_x = torch.tensor([[1, 1], [2, 2], [3, 3]])
        expected_y = torch.tensor([[4, 5], [4, 5], [4, 5]])

        torch.testing.assert_close(grid_x, expected_x)
        torch.testing.assert_close(grid_y, expected_y)

    def test_output_matches_torch_meshgrid_ij(self):
        """Output matches torch.meshgrid with indexing='ij'."""
        a = torch.arange(4, dtype=torch.float32)
        b = torch.arange(5, dtype=torch.float32)

        result = meshgrid(a, b)
        expected = torch.meshgrid(a, b, indexing="ij")

        for r, e in zip(result, expected):
            torch.testing.assert_close(r, e)
