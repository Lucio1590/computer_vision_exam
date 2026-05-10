"""
Visualization utilities.

Renders bounding boxes, labels, and metric plots. Follows KISS principle:
only consumes validated NMS outputs.
"""

import cv2
import numpy as np
from src.utils.config import VISUALIZATION, get_coco_classes


def draw_detections(image, detections, class_names=None):
    """
    Draw bounding boxes and class labels on an image.

    Args:
        image: BGR image (numpy array).
        detections: List of [class_id, confidence, x, y, w, h].
        class_names: Optional list of class names. Loads COCO names if None.

    Returns:
        Annotated image.
    """
    if class_names is None:
        class_names = get_coco_classes()

    img_h, img_w = image.shape[:2]
    out = image.copy()

    for det in detections:
        class_id, conf, x, y, w, h = det
        class_id = int(class_id)

        # Convert normalized coordinates to absolute if needed
        if max(x, y, w, h) <= 1.0:
            x, y, w, h = int(x * img_w), int(y * img_h), int(w * img_w), int(h * img_h)
        else:
            x, y, w, h = int(x), int(y), int(w), int(h)

        label = f"{class_names[class_id]}: {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(
            out,
            (x, y),
            (x + w, y + h),
            VISUALIZATION["box_color"],
            VISUALIZATION["thickness"],
        )

        # Draw label background
        (tw, th), _ = cv2.getTextSize(
            label, VISUALIZATION["font"], VISUALIZATION["font_scale"], 1
        )
        cv2.rectangle(
            out,
            (x, y - th - 4),
            (x + tw, y),
            VISUALIZATION["box_color"],
            -1,
        )

        # Draw label text
        cv2.putText(
            out,
            label,
            (x, y - 2),
            VISUALIZATION["font"],
            VISUALIZATION["font_scale"],
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA,
        )

    return out
