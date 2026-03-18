import numpy as np
import pytest
import torch

from yolox.utils.boxes import (
    adjust_box_anns,
    bboxes_iou,
    cxcywh2xyxy,
    filter_box,
    matrix_iou,
    postprocess,
    xyxy2cxcywh,
    xyxy2xywh,
)


class TestBboxesIou:
    def test_identical_boxes_returns_iou_one(self):
        boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 100.0, 100.0]])
        iou = bboxes_iou(boxes, boxes, xyxy=True)
        assert iou.shape == (2, 2)
        assert torch.allclose(iou.diag(), torch.ones(2))

    def test_non_overlapping_boxes_returns_iou_zero(self):
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        iou = bboxes_iou(a, b, xyxy=True)
        assert iou.item() == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        b = torch.tensor([[5.0, 5.0, 15.0, 15.0]])
        iou = bboxes_iou(a, b, xyxy=True)
        # Intersection: 5x5=25, Union: 100+100-25=175
        assert iou.item() == pytest.approx(25.0 / 175.0, abs=1e-4)

    def test_xyxy_false_cxcywh_format(self):
        # cxcywh: center_x, center_y, w, h
        # Box 1: center (5,5) w=10 h=10 -> xyxy (0,0,10,10)
        # Box 2: center (5,5) w=10 h=10 -> same box
        a = torch.tensor([[5.0, 5.0, 10.0, 10.0]])
        b = torch.tensor([[5.0, 5.0, 10.0, 10.0]])
        iou = bboxes_iou(a, b, xyxy=False)
        assert iou.item() == pytest.approx(1.0, abs=1e-4)


class TestMatrixIou:
    def test_identical_boxes_returns_iou_one(self):
        boxes = np.array([[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 100.0, 100.0]])
        iou = matrix_iou(boxes, boxes)
        assert iou.shape == (2, 2)
        np.testing.assert_allclose(np.diag(iou), 1.0, atol=1e-5)

    def test_non_overlapping_boxes_returns_iou_zero(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[20.0, 20.0, 30.0, 30.0]])
        iou = matrix_iou(a, b)
        assert iou.item() == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[5.0, 5.0, 15.0, 15.0]])
        iou = matrix_iou(a, b)
        assert iou.item() == pytest.approx(25.0 / 175.0, abs=1e-4)


class TestXyxy2Xywh:
    def test_conversion(self):
        bboxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
        result = xyxy2xywh(bboxes.clone())
        # x1 stays, y1 stays, w = x2-x1 = 40, h = y2-y1 = 60
        expected = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
        assert torch.allclose(result, expected)


class TestXyxy2Cxcywh:
    def test_conversion(self):
        bboxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
        result = xyxy2cxcywh(bboxes.clone())
        # cx = (10+50)/2=30, cy = (20+80)/2=50, w=40, h=60
        expected = torch.tensor([[30.0, 50.0, 40.0, 60.0]])
        assert torch.allclose(result, expected)


class TestCxcywh2Xyxy:
    def test_actual_behavior(self):
        """Test the actual (buggy) behavior of cxcywh2xyxy.

        The function modifies values in-place, leading to incorrect results
        because it uses already-modified x0/y0 when computing x2/y2.
        Specifically:
          x0 = cx - w/2  (x0 is now modified)
          y0 = cy - h/2  (y0 is now modified)
          x2 = x0 + w    (uses modified x0 = cx - w/2, so x2 = cx + w/2) -- actually correct
          y2 = y0 + h    (uses modified y0 = cy - h/2, so y2 = cy + h/2) -- actually correct
        So the bug may produce correct results in some implementations.
        We test actual output rather than assumed output.
        """
        bboxes = torch.tensor([[30.0, 50.0, 40.0, 60.0]])
        result = cxcywh2xyxy(bboxes.clone())
        # Record actual behavior: run and snapshot
        # If x0 = cx - w/2 = 30 - 20 = 10, y0 = 50 - 30 = 20
        # x2 = x0 + w = 10 + 40 = 50, y2 = y0 + h = 20 + 60 = 80
        # This would be correct. But the note says there's a bug with
        # already-modified values. We test what actually happens.
        # The in-place operation: bboxes[:,0] = cx - w/2, then bboxes[:,2] = bboxes[:,0] + w
        # bboxes[:,0] is already cx - w/2, so bboxes[:,2] = cx - w/2 + w = cx + w/2 -- correct!
        # So the result is actually correct for this case.
        expected = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
        assert torch.allclose(result, expected)


class TestPostprocess:
    def _make_prediction(self, conf, num_classes=80):
        """Create a (1, N, 5+num_classes) prediction tensor in cxcywh format."""
        pred = torch.zeros(1, 2, 5 + num_classes)
        # Detection 0: valid box
        pred[0, 0, 0] = 50.0   # cx
        pred[0, 0, 1] = 50.0   # cy
        pred[0, 0, 2] = 20.0   # w
        pred[0, 0, 3] = 20.0   # h
        pred[0, 0, 4] = conf    # objectness
        pred[0, 0, 5] = conf    # class 0 confidence
        # Detection 1: low confidence
        pred[0, 1, 0] = 100.0
        pred[0, 1, 1] = 100.0
        pred[0, 1, 2] = 10.0
        pred[0, 1, 3] = 10.0
        pred[0, 1, 4] = 0.001
        pred[0, 1, 5] = 0.001
        return pred

    def test_high_confidence_survives(self):
        pred = self._make_prediction(0.95, num_classes=80)
        results = postprocess(pred, num_classes=80, conf_thre=0.7, nms_thre=0.45)
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].shape[0] >= 1

    def test_low_confidence_filtered(self):
        pred = self._make_prediction(0.1, num_classes=80)
        results = postprocess(pred, num_classes=80, conf_thre=0.7, nms_thre=0.45)
        assert len(results) == 1
        assert results[0] is None

    def test_empty_input_returns_nones(self):
        pred = torch.zeros(2, 0, 85)
        results = postprocess(pred, num_classes=80, conf_thre=0.7, nms_thre=0.45)
        assert len(results) == 2
        assert all(r is None for r in results)


class TestFilterBox:
    def test_filters_by_area_range(self):
        # filter_box: keep where w*h > min_scale^2 AND w*h < max_scale^2
        # scale_range=(3, 50): keep area > 9 and area < 2500
        output = torch.tensor(
            [
                [5.0, 5.0, 10.0, 10.0, 0.9, 0.9, 0.0],   # w=5, h=5, area=25 -> kept
                [0.0, 0.0, 100.0, 100.0, 0.9, 0.9, 0.0],  # w=100, h=100, area=10000 -> filtered
                [0.0, 0.0, 4.0, 4.0, 0.9, 0.9, 0.0],      # w=4, h=4, area=16 -> kept
            ]
        )
        result = filter_box(output, scale_range=(3, 50))
        assert result.shape[0] == 2


class TestAdjustBoxAnns:
    def test_scaling_and_clamping(self):
        bbox = np.array([[10.0, 20.0, 50.0, 80.0]])
        result = adjust_box_anns(bbox, scale_ratio=2.0, padw=5, padh=10, w_max=120, h_max=180)
        # Expected: bbox * 2.0 + pad, then clip
        # x1 = 10*2+5=25, y1=20*2+10=50, x2=50*2+5=105, y2=80*2+10=170
        expected = np.array([[25.0, 50.0, 105.0, 170.0]])
        np.testing.assert_allclose(result, expected)

    def test_clamping_to_bounds(self):
        bbox = np.array([[10.0, 20.0, 100.0, 100.0]])
        result = adjust_box_anns(bbox, scale_ratio=2.0, padw=0, padh=0, w_max=150, h_max=150)
        # x1=20, y1=40, x2=min(200,150)=150, y2=min(200,150)=150
        expected = np.array([[20.0, 40.0, 150.0, 150.0]])
        np.testing.assert_allclose(result, expected)
