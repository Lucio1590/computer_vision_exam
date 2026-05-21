"""
Side-by-side comparison of Classical vs Deep Learning gesture pipelines.

Evaluates both pipelines on test images / datasets and generates
a comparison report with detection metrics, classification accuracy,
and inference speed.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import GESTURE_CLASSES, YOLO_HAND_MODEL_PATH, GESTURE_CNN_PATH, SVM_KEYPOINTS_PATH
from src.classical.hand_detector import HandDetector
from src.classical.feature_extractor import FeatureExtractor
from src.classical.gesture_classifier import GestureClassifier
from src.deep_learning.yolo_detector import YOLOHandDetector
from src.deep_learning.gesture_net import GestureNet
from src.evaluation.benchmark import benchmark_detector
from src.evaluation.metrics import plot_confusion_matrix
from src.utils.visualization import draw_detections


class ClassicalPipeline:
    """End-to-end classical pipeline: MediaPipe -> features -> SVM."""

    def __init__(self):
        self.detector = HandDetector()
        self.extractor = FeatureExtractor()
        self.classifier = GestureClassifier()
        self.classifier.load(str(SVM_KEYPOINTS_PATH))

    def detect(self, image: np.ndarray):
        """Return detections in [class_id, confidence, x, y, w, h] format."""
        hands = self.detector.detect(image)
        detections = []
        for landmarks in hands:
            features = self.extractor.extract(landmarks)
            pred_class, conf = self.classifier.predict(features)

            # Compute bbox from landmarks (21 points, normalized 0-1)
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]
            h_img, w_img = image.shape[:2]
            x_min = int(min(xs) * w_img)
            y_min = int(min(ys) * h_img)
            x_max = int(max(xs) * w_img)
            y_max = int(max(ys) * h_img)

            detections.append([
                int(pred_class),
                float(conf),
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min,
            ])
        return detections


class DeepPipeline:
    """End-to-end deep learning pipeline: YOLO -> crop -> CNN."""

    def __init__(self, device=None):
        self.detector = YOLOHandDetector(str(YOLO_HAND_MODEL_PATH))
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.classifier = GestureNet(num_classes=len(GESTURE_CLASSES))
        self.classifier.load_state_dict(torch.load(GESTURE_CNN_PATH, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()

        self.img_size = 224

    def detect(self, image: np.ndarray):
        """Return detections in [class_id, confidence, x, y, w, h] format."""
        import torchvision.transforms as T

        hands = self.detector.detect(image)
        detections = []

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for det in hands:
            _, conf_det, x, y, w, h = det
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            tensor = transform(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.classifier(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred_class = int(torch.argmax(probs))
                conf = float(probs[pred_class])

            detections.append([pred_class, conf * conf_det, x, y, w, h])
        return detections


def evaluate_on_images(classical: ClassicalPipeline, deep: DeepPipeline, image_paths, output_dir: Path):
    """Run both pipelines on a list of images and collect detection + timing metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    valid_paths = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
            valid_paths.append(p)

    if len(images) == 0:
        raise ValueError("No valid images provided for comparison.")

    # Benchmark detection speed
    print("Benchmarking classical pipeline...")
    classical_bench = benchmark_detector(classical, images, warmup=1)

    print("Benchmarking deep learning pipeline...")
    deep_bench = benchmark_detector(deep, images, warmup=1)

    classical_results = []
    deep_results = []

    for i, image in enumerate(images):
        c_dets = classical.detect(image)
        d_dets = deep.detect(image)

        classical_results.append(c_dets)
        deep_results.append(d_dets)

        # Save annotated images
        c_vis = draw_detections(image.copy(), c_dets, class_names=GESTURE_CLASSES)
        d_vis = draw_detections(image.copy(), d_dets, class_names=GESTURE_CLASSES)

        cv2.imwrite(str(output_dir / f"classical_{i:03d}.jpg"), c_vis)
        cv2.imwrite(str(output_dir / f"deep_{i:03d}.jpg"), d_vis)

    report = {
        "num_images": len(images),
        "classical": {
            "fps": classical_bench["mean_fps"],
            "latency_ms": classical_bench["mean_latency_ms"],
            "total_detections": sum(len(r) for r in classical_results),
        },
        "deep": {
            "fps": deep_bench["mean_fps"],
            "latency_ms": deep_bench["mean_latency_ms"],
            "total_detections": sum(len(r) for r in deep_results),
        },
    }

    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Comparison complete. Results saved to {output_dir}")
    return report


def evaluate_classifier_dataset(classical: ClassicalPipeline, deep: DeepPipeline, data_dir: Path, output_dir: Path):
    """Evaluate both pipelines on the classifier crop dataset (ground-truth labels available)."""
    from torchvision import datasets
    import torchvision.transforms as T

    output_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    # Use ImageFolder to get ground-truth labels
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    y_true = []
    y_classical = []
    y_deep = []

    print(f"Evaluating on {len(dataset)} samples from {data_dir}...")

    for idx in range(len(dataset)):
        tensor, label = dataset[idx]
        img_path = dataset.samples[idx][0]
        image = cv2.imread(img_path)
        if image is None:
            continue

        y_true.append(label)

        # Classical
        c_dets = classical.detect(image)
        if len(c_dets) > 0:
            # Take highest confidence detection
            c_dets.sort(key=lambda d: d[1], reverse=True)
            y_classical.append(c_dets[0][0])
        else:
            y_classical.append(-1)  # No detection

        # Deep
        d_dets = deep.detect(image)
        if len(d_dets) > 0:
            d_dets.sort(key=lambda d: d[1], reverse=True)
            y_deep.append(d_dets[0][0])
        else:
            y_deep.append(-1)

    # Filter out no-detections for metric calculation
    valid_classical = [(t, p) for t, p in zip(y_true, y_classical) if p != -1]
    valid_deep = [(t, p) for t, p in zip(y_true, y_deep) if p != -1]

    def _metrics(y_t, y_p):
        return {
            "accuracy": accuracy_score(y_t, y_p),
            "precision": precision_score(y_t, y_p, average="macro", zero_division=0),
            "recall": recall_score(y_t, y_p, average="macro", zero_division=0),
            "f1": f1_score(y_t, y_p, average="macro", zero_division=0),
        }

    metrics = {
        "num_samples": len(y_true),
        "classical": {
            **_metrics([v[0] for v in valid_classical], [v[1] for v in valid_classical]),
            "detection_rate": len(valid_classical) / len(y_true),
        },
        "deep": {
            **_metrics([v[0] for v in valid_deep], [v[1] for v in valid_deep]),
            "detection_rate": len(valid_deep) / len(y_true),
        },
    }

    with open(output_dir / "classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrices
    if len(valid_classical) > 0:
        plot_confusion_matrix(
            [v[0] for v in valid_classical],
            [v[1] for v in valid_classical],
            GESTURE_CLASSES,
            save_path=output_dir / "confusion_classical.jpg",
        )
    if len(valid_deep) > 0:
        plot_confusion_matrix(
            [v[0] for v in valid_deep],
            [v[1] for v in valid_deep],
            GESTURE_CLASSES,
            save_path=output_dir / "confusion_deep.jpg",
        )

    print("Classification evaluation complete.")
    print(f"  Classical accuracy: {metrics['classical']['accuracy']:.4f} (detected {metrics['classical']['detection_rate']:.2%})")
    print(f"  Deep accuracy:      {metrics['deep']['accuracy']:.4f} (detected {metrics['deep']['detection_rate']:.2%})")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare Classical vs Deep Learning gesture pipelines")
    parser.add_argument("--mode", choices=["images", "dataset"], default="images",
                        help="Evaluation mode: 'images' for raw images, 'dataset' for classifier test set")
    parser.add_argument("--images", nargs="*", help="Paths to test images (mode=images)")
    parser.add_argument("--dataset", default="data/processed/classifier/test",
                        help="Path to classifier test dataset (mode=dataset)")
    parser.add_argument("--output", default="docs/results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading classical pipeline...")
    classical = ClassicalPipeline()

    print("Loading deep learning pipeline...")
    deep = DeepPipeline()

    if args.mode == "images":
        if not args.images:
            raise ValueError("--images required when mode=images")
        image_paths = [Path(p) for p in args.images]
        report = evaluate_on_images(classical, deep, image_paths, output_dir)
        print("\n--- Detection Comparison Report ---")
        print(f"Images processed: {report['num_images']}")
        print(f"Classical FPS: {report['classical']['fps']:.2f}  |  Detections: {report['classical']['total_detections']}")
        print(f"Deep FPS:      {report['deep']['fps']:.2f}  |  Detections: {report['deep']['total_detections']}")
    else:
        metrics = evaluate_classifier_dataset(classical, deep, Path(args.dataset), output_dir)
        print("\n--- Classification Comparison Report ---")
        print(f"Classical: accuracy={metrics['classical']['accuracy']:.4f}, precision={metrics['classical']['precision']:.4f}, "
              f"recall={metrics['classical']['recall']:.4f}, f1={metrics['classical']['f1']:.4f}")
        print(f"Deep:      accuracy={metrics['deep']['accuracy']:.4f}, precision={metrics['deep']['precision']:.4f}, "
              f"recall={metrics['deep']['recall']:.4f}, f1={metrics['deep']['f1']:.4f}")


if __name__ == "__main__":
    main()
