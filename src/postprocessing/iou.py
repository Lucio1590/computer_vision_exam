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
        format: "xywh" (center x,y + w,h) or "xyxy" (min/max corners).

    Returns:
        IoU value in range [0.0, 1.0].
    """
    # TODO: Implement IoU calculation
    # 1. Convert both boxes to xyxy if needed
    # 2. Compute intersection coordinates
    # 3. Compute intersection area
    # 4. Compute union area = area_a + area_b - intersection_area
    # 5. Return intersection_area / union_area
    pass


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
    # TODO: Implement pairwise IoU matrix (useful for mAP calculation)
    pass
