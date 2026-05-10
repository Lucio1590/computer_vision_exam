"""
Evaluation metrics for object detection.

Implements detection-specific metrics (mAP, IoU) and classification
metrics (Precision, Recall, F1, Confusion Matrix) as required by the exam.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def compute_precision_recall_f1(y_true, y_pred, average="macro"):
    """
    Compute classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy ('macro', 'micro', 'weighted').

    Returns:
        Dictionary with precision, recall, f1.
    """
    return {
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def compute_ap(recalls, precisions):
    """
    Compute Average Precision using the all-point interpolation method.

    Args:
        recalls: Array of recall values.
        precisions: Array of precision values.

    Returns:
        Average Precision (float).
    """
    # TODO: Implement AP calculation
    # 1. Sort by recall
    # 2. Interpolate precision (make it monotonically decreasing)
    # 3. Compute area under PR curve
    pass


def compute_map(predictions, ground_truths, iou_threshold=0.5, num_classes=80):
    """
    Compute mean Average Precision (mAP) across all classes.

    Args:
        predictions: List of predictions per image.
        ground_truths: List of ground-truth annotations per image.
        iou_threshold: IoU threshold for matching.
        num_classes: Number of classes.

    Returns:
        mAP value.
    """
    # TODO: Implement mAP calculation
    # 1. Group predictions and GTs by class
    # 2. For each class, sort predictions by confidence
    # 3. Match predictions to GTs using IoU > threshold
    # 4. Compute TP/FP for each confidence level
    # 5. Calculate AP for the class
    # 6. Average AP across classes
    pass


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save figure.
    """
    # TODO: Use seaborn heatmap for visualization
    pass
