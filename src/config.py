"""
Centralized configuration for the Hand Gesture Recognition Framework.

Single source of truth for paths, class names, and hyperparameters.
Import this module instead of hard-coding values.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

# Processed subdirectories
YOLO_DATASET_DIR = PROCESSED_DATA_DIR / "yolo"
CLASSIFIER_DATASET_DIR = PROCESSED_DATA_DIR / "classifier"
KEYPOINTS_DATASET_DIR = PROCESSED_DATA_DIR / "keypoints"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_HAND_MODEL_PATH = MODELS_DIR / "yolov8n_hand.pt"
YOLO_HAND_MODEL_FALLBACK = (
    PROJECT_ROOT / "runs" / "detect" / "models" / "yolov8n_hand" / "hand_detector" / "weights" / "best.pt"
)
GESTURE_CNN_PATH = MODELS_DIR / "gesture_cnn.pt"
SVM_KEYPOINTS_PATH = MODELS_DIR / "svm_keypoints.pkl"

# Gesture classes (HaGRID subset)
GESTURE_CLASSES = ["dislike", "fist", "like", "ok", "palm", "peace"]
NUM_GESTURE_CLASSES = len(GESTURE_CLASSES)

# Ensure directories exist
for d in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ANNOTATIONS_DIR,
    YOLO_DATASET_DIR,
    CLASSIFIER_DATASET_DIR,
    KEYPOINTS_DATASET_DIR,
    MODELS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# Preprocessing parameters
PREPROCESSING = {
    "yolo_input_size": 640,
    "classifier_input_size": 224,
    "normalize_mean": [0.485, 0.456, 0.406],  # ImageNet
    "normalize_std": [0.229, 0.224, 0.225],   # ImageNet
}

# Classical pipeline parameters
CLASSICAL = {
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.5,
    "svm_kernel": "rbf",
    "svm_c": 1.0,
}

# YOLO parameters
YOLO = {
    "model_name": "yolov8n.pt",
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.01,
    "conf_threshold": 0.5,
    "nms_threshold": 0.4,
    "single_class": True,  # Only "hand"
}

# CNN parameters
CNN = {
    "architecture": "resnet18",  # or "custom"
    "num_classes": NUM_GESTURE_CLASSES,
    "epochs": 50,
    "batch_size": 32,
    "lr": 0.001,
    "patience": 5,
}

# Evaluation parameters
EVALUATION = {
    "iou_threshold": 0.5,
    "map_iou_thresholds": [0.5, 0.75],
    "gesture_classes": GESTURE_CLASSES,
}

# Visualization parameters
VISUALIZATION = {
    "box_color": (0, 255, 0),
    "text_color": (0, 0, 255),
    "font": 0,
    "font_scale": 0.5,
    "thickness": 2,
    "keypoint_color": (0, 255, 0),
    "skeleton_color": (255, 0, 0),
}

# Webcam parameters
WEBCAM = {
    "source": 0,
    "width": 1280,
    "height": 720,
    "fps": 30,
}
