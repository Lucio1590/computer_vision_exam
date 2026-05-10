"""
Unit tests for IoU computation.
"""

import numpy as np
from src.postprocessing.iou import compute_iou, compute_iou_matrix


def test_iou_perfect_overlap():
    box = [0, 0, 10, 10]
    assert compute_iou(box, box, format="xywh") == 1.0


def test_iou_no_overlap():
    box_a = [0, 0, 10, 10]
    box_b = [20, 20, 10, 10]
    assert compute_iou(box_a, box_b, format="xywh") == 0.0


def test_iou_partial_overlap():
    box_a = [0, 0, 10, 10]
    box_b = [5, 5, 10, 10]
    # Intersection = 5x5 = 25, Union = 100 + 100 - 25 = 175
    expected = 25.0 / 175.0
    result = compute_iou(box_a, box_b, format="xywh")
    assert abs(result - expected) < 1e-6


def test_iou_xyxy_format():
    box_a = [0, 0, 10, 10]
    box_b = [5, 5, 15, 15]
    expected = 25.0 / 175.0
    result = compute_iou(box_a, box_b, format="xyxy")
    assert abs(result - expected) < 1e-6


def test_iou_matrix():
    boxes_a = [[0, 0, 10, 10], [20, 20, 10, 10]]
    boxes_b = [[0, 0, 10, 10], [5, 5, 10, 10]]
    matrix = compute_iou_matrix(boxes_a, boxes_b, format="xywh")
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.0
    assert matrix[0, 1] == 25.0 / 175.0
    assert matrix[1, 0] == 0.0
