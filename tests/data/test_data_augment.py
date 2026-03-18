"""Tests for yolox/data/data_augment.py."""

import numpy as np
import pytest

from yolox.data.data_augment import TrainTransform, ValTransform, augment_hsv, preproc


class TestPreproc:
    """Tests for preproc function."""

    def test_output_shape(self):
        """Output shape is (3, H, W) for input_size=(H, W)."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        input_size = (416, 416)

        padded_img, ratio = preproc(img, input_size)

        assert padded_img.shape == (3, 416, 416)
        assert padded_img.dtype == np.float32

    def test_padding_value_is_114(self):
        """Padding region is filled with 114."""
        # Small image that will need significant padding
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        input_size = (416, 416)

        padded_img, ratio = preproc(img, input_size)

        # The padded region (bottom-right) should be 114
        # After swap, shape is (3, H, W). The image is placed top-left.
        resized_h = int(100 * ratio)
        resized_w = int(100 * ratio)
        # Check padding area (channels, bottom rows)
        pad_region = padded_img[:, resized_h + 1 :, :]
        if pad_region.size > 0:
            assert np.all(
                pad_region == 114.0
            ), "Padding region should be filled with 114"

    def test_aspect_ratio_preserved(self):
        """Aspect ratio is preserved and ratio is correct."""
        img = np.random.randint(0, 255, (300, 600, 3), dtype=np.uint8)
        input_size = (416, 416)

        padded_img, ratio = preproc(img, input_size)

        # ratio should be min(target_h/img_h, target_w/img_w)
        expected_ratio = min(416 / 300, 416 / 600)
        assert abs(ratio - expected_ratio) < 1e-6


class TestAugmentHsv:
    """Tests for augment_hsv function."""

    def test_shape_unchanged_and_valid_range(self):
        """Image shape unchanged and values in valid range after HSV augmentation."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        original_shape = img.shape

        augment_hsv(img)

        assert img.shape == original_shape
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255


class TestTrainTransform:
    """Tests for TrainTransform."""

    def test_output_shapes(self):
        """Output shapes are correct."""
        transform = TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0)

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # targets: each row is [cls, x1, y1, x2, y2]
        targets = np.array(
            [[0, 100, 100, 200, 200], [1, 300, 300, 400, 400]], dtype=np.float32
        )
        input_dim = (416, 416)

        out_img, padded_labels = transform(img, targets, input_dim)

        assert out_img.shape == (3, 416, 416)
        assert out_img.dtype == np.float32
        assert padded_labels.shape == (50, 5)

    def test_with_empty_targets(self):
        """Handles empty targets (no bounding boxes)."""
        transform = TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0)

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        targets = np.zeros((0, 5), dtype=np.float32)
        input_dim = (416, 416)

        out_img, padded_labels = transform(img, targets, input_dim)

        assert out_img.shape == (3, 416, 416)
        assert padded_labels.shape == (50, 5)
        # All labels should be zeros when there are no targets
        assert np.all(padded_labels == 0)


class TestValTransform:
    """Tests for ValTransform."""

    def test_output_shape(self):
        """Output shape is correct."""
        transform = ValTransform()

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        input_size = (416, 416)

        out_img, out_labels = transform(img, None, input_size)

        assert out_img.shape == (3, 416, 416)
        assert out_img.dtype == np.float32
