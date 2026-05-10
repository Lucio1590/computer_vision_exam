"""
Unit tests for Non-Maximum Suppression.
"""

from src.postprocessing.nms import manual_nms, opencv_nms


def test_manual_nms_identical_boxes():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    scores = [0.9, 0.8]
    class_ids = [0, 0]
    keep = manual_nms(boxes, scores, class_ids, iou_threshold=0.5)
    assert len(keep) == 1
    assert keep[0] == 0  # Higher score kept


def test_manual_nms_low_iou():
    boxes = [[0, 0, 10, 10], [20, 20, 10, 10]]
    scores = [0.9, 0.8]
    class_ids = [0, 0]
    keep = manual_nms(boxes, scores, class_ids, iou_threshold=0.5)
    assert len(keep) == 2


def test_manual_nms_different_classes():
    boxes = [[0, 0, 10, 10], [1, 1, 10, 10]]
    scores = [0.9, 0.85]
    class_ids = [0, 1]
    keep = manual_nms(boxes, scores, class_ids, iou_threshold=0.5)
    assert len(keep) == 2  # Different classes should not suppress


def test_opencv_nms_empty():
    keep = opencv_nms([], [], 0.5, 0.4)
    assert keep == []


def test_opencv_nms_basic():
    boxes = [[0, 0, 10, 10], [20, 20, 10, 10]]
    scores = [0.9, 0.8]
    keep = opencv_nms(boxes, scores, 0.5, 0.4)
    assert len(keep) == 2
