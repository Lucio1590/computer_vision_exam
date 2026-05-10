"""
Model loading utilities (DRY principle).

Centralizes cv2.dnn.readNet calls to prevent redundant I/O and ensure
consistent initialization across the codebase.
"""

import cv2
from src.utils.config import YOLO_WEIGHTS_PATH, YOLO_CFG_PATH


def load_yolo_network():
    """
    Load YOLO network from weights and config files.

    Returns:
        cv2.dnn.Net instance.
    """
    if not YOLO_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {YOLO_WEIGHTS_PATH}. "
            "Run 'python scripts/download_models.py' to download them."
        )
    if not YOLO_CFG_PATH.exists():
        raise FileNotFoundError(
            f"YOLO config not found at {YOLO_CFG_PATH}. "
            "Run 'python scripts/download_models.py' to download it."
        )

    net = cv2.dnn.readNet(
        str(YOLO_WEIGHTS_PATH),
        str(YOLO_CFG_PATH)
    )
    # Optional: enable CUDA backend if available
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net
