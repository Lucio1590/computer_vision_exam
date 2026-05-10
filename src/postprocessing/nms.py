"""
Non-Maximum Suppression (NMS) implementation.

Includes both a manual (educational) implementation and a wrapper
around cv2.dnn.NMSBoxes for production speed.
"""

import cv2
import numpy as np
from src.postprocessing.iou import compute_iou


def manual_nms(boxes, scores, class_ids, iou_threshold=0.5):
    """
    Manual NMS algorithm for educational transparency.

    Algorithm:
        1. Sort boxes by confidence score (descending).
        2. Select the box with highest score (M) as a local maximum.
        3. Suppress all remaining boxes with IoU(M, box) > threshold.
        4. Repeat steps 2-3 with remaining boxes until list is empty.

    Args:
        boxes: List of boxes in [x, y, w, h] format.
        scores: List of confidence scores.
        class_ids: List of class IDs.
        iou_threshold: IoU threshold for suppression.

    Returns:
        List of indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(int(current))

        if len(indices) == 1:
            break

        current_box = boxes[current]
        remaining_boxes = [boxes[i] for i in indices[1:]]

        ious = [compute_iou(current_box, b, format="xywh") for b in remaining_boxes]

        # Keep boxes with IoU below threshold OR different class
        mask = [
            (ious[i] < iou_threshold) or (class_ids[indices[i + 1]] != class_ids[current])
            for i in range(len(ious))
        ]
        indices = indices[1:][mask]

    return keep


def opencv_nms(boxes, scores, conf_threshold=0.5, nms_threshold=0.4):
    """
    Production-grade NMS using OpenCV's optimized implementation.

    Args:
        boxes: List of boxes in [x, y, w, h] format.
        scores: List of confidence scores.
        conf_threshold: Minimum confidence to consider.
        nms_threshold: IoU threshold for suppression.

    Returns:
        List of indices of boxes to keep.
    """
    if len(boxes) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    if indices is None:
        return []
    # Flatten nested arrays to plain Python ints
    return [int(i) for i in indices.flatten()]
