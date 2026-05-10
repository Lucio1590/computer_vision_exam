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
    # TODO: Implement manual NMS
    # Hint: Use numpy.argsort, then iterate and compute IoU with kept boxes.
    pass


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
    # TODO: Implement cv2.dnn.NMSBoxes wrapper
    # indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    pass
