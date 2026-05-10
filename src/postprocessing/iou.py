"""
Intersection over Union (IoU) computation.

Provides mathematical transparency for bounding box overlap measurement.
Supports both [x_min, y_min, x_max, y_max] and [x, y, w, h] formats.
"""

import numpy as np


def convert_xywh_to_xyxy(box):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def compute_iou(box_a, box_b, format="xywh") -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Formula: IoU = Area(Intersection) / Area(Union)

    Args:
        box_a: First box.
        box_b: Second box.
        format: "xywh" (top-left x,y + w,h) or "xyxy" (min/max corners).

    Returns:
        IoU value in range [0.0, 1.0].
    """
    if format == "xywh":
        box_a = convert_xywh_to_xyxy(box_a)
        box_b = convert_xywh_to_xyxy(box_b)

    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_iou_matrix(boxes_a: list, boxes_b: list, format="xywh") -> np.ndarray:
    """
    Compute pairwise IoU matrix between two lists of boxes.

    Args:
        boxes_a: List of N boxes.
        boxes_b: List of M boxes.
        format: Box format string.

    Returns:
        N x M numpy array of IoU values.
    """
    n = len(boxes_a)
    m = len(boxes_b)
    matrix = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            matrix[i, j] = compute_iou(boxes_a[i], boxes_b[j], format=format)
    return matrix
