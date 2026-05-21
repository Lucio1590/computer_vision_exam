"""
Centralized configuration handler for the object detection framework.

This module ensures DRY (Don't Repeat Yourself) by providing a single source
of truth for paths, hyperparameters, and network settings.

NOTE: Prefer importing from src.config for new code. This file is kept for
backward compatibility with legacy modules.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Legacy paths (kept for backward compatibility)
YOLO_WEIGHTS_PATH = MODELS_DIR / "yolov3.weights"
YOLO_CFG_PATH = MODELS_DIR / "yolov3.cfg"
COCO_NAMES_PATH = MODELS_DIR / "coco.names"
HOG_SVM_PATH = MODELS_DIR / "hog_svm.pkl"

# Ensure directories exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ANNOTATIONS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Preprocessing parameters
PREPROCESSING = {
    "input_size": (416, 416),      # Width, Height for legacy YOLO
    "scalefactor": 1 / 255.0,       # Normalization factor
    "mean": (0, 0, 0),              # Mean subtraction values (BGR)
    "swap_rb": True,                # BGR -> RGB swap
    "crop": False,                  # Whether to crop after resize
}

# Legacy YOLO inference parameters
YOLO = {
    "conf_threshold": 0.5,
    "nms_threshold": 0.4,
    "output_format": "xywh",        # [x_center, y_center, width, height]
}

# Legacy Classical HOG+SVM parameters
CLASSICAL = {
    "hog_win_size": (64, 128),      # Width, Height
    "hog_block_size": (16, 16),
    "hog_block_stride": (8, 8),
    "hog_cell_size": (8, 8),
    "hog_nbins": 9,
    "svm_kernel": "linear",
    "svm_c": 0.01,
    "scale_factor": 1.05,           # Image pyramid scale
    "min_neighbors": 3,             # Min neighbors for grouping (custom NMS-like)
}

# Evaluation parameters
EVALUATION = {
    "iou_threshold": 0.5,
    "map_iou_thresholds": [0.5, 0.75],
}

# Visualization parameters
VISUALIZATION = {
    "box_color": (0, 255, 0),       # BGR green
    "text_color": (0, 0, 255),      # BGR red
    "font": 0,                      # cv2.FONT_HERSHEY_SIMPLEX
    "font_scale": 0.5,
    "thickness": 2,
}


def get_coco_classes() -> list:
    """Load COCO class names from file."""
    if not COCO_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"coco.names not found at {COCO_NAMES_PATH}. "
            "Run 'python scripts/download_models.py' first."
        )
    with open(COCO_NAMES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]
