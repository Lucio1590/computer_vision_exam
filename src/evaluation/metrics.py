"""
Evaluation metrics for object detection.

Implements detection-specific metrics (mAP, IoU) and classification
metrics (Precision, Recall, F1, Confusion Matrix) as required by the exam.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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
    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # Interpolate precision: make it monotonically decreasing
    # Start from the end and take cumulative maximum
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Add sentinel values at start and end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([precisions[0]] if len(precisions) > 0 else [0.0], precisions, [0.0]))

    # Compute area under PR curve
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return float(ap)


def compute_map(predictions, ground_truths, iou_threshold=0.5, num_classes=80):
    """
    Compute mean Average Precision (mAP) across all classes.

    Args:
        predictions: List of dicts per image, each with 'boxes': [[x,y,w,h],...],
                     'scores': [...], 'class_ids': [...].
        ground_truths: List of dicts per image, each with 'boxes': [[x,y,w,h],...],
                       'class_ids': [...].
        iou_threshold: IoU threshold for matching.
        num_classes: Number of classes.

    Returns:
        mAP value.
    """
    from src.postprocessing.iou import compute_iou

    aps = []

    for class_id in range(num_classes):
        # Gather all predictions and GTs for this class across images
        class_preds = []
        class_gts = []

        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            pred_mask = [i for i, cid in enumerate(pred["class_ids"]) if cid == class_id]
            gt_mask = [i for i, cid in enumerate(gt["class_ids"]) if cid == class_id]

            for i in pred_mask:
                class_preds.append({
                    "image_id": img_idx,
                    "box": pred["boxes"][i],
                    "score": pred["scores"][i],
                    "matched": False,
                })

            for i in gt_mask:
                class_gts.append({
                    "image_id": img_idx,
                    "box": gt["boxes"][i],
                    "matched": False,
                })

        if len(class_preds) == 0:
            continue

        # Sort predictions by confidence descending
        class_preds.sort(key=lambda p: p["score"], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for pred_idx, pred in enumerate(class_preds):
            # Find unmatched GTs in same image
            gt_candidates = [
                gt for gt in class_gts
                if gt["image_id"] == pred["image_id"] and not gt["matched"]
            ]

            if len(gt_candidates) == 0:
                fp[pred_idx] = 1
                continue

            # Compute IoU with all candidates
            ious = [compute_iou(pred["box"], gt["box"], format="xywh") for gt in gt_candidates]
            best_idx = int(np.argmax(ious))
            best_iou = ious[best_idx]

            if best_iou >= iou_threshold:
                tp[pred_idx] = 1
                # Mark GT as matched
                gt_candidates[best_idx]["matched"] = True
                # Update original list
                for gt in class_gts:
                    if gt["image_id"] == pred["image_id"] and gt["box"] == gt_candidates[best_idx]["box"]:
                        gt["matched"] = True
            else:
                fp[pred_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / max(len(class_gts), 1)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
