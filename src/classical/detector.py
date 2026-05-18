"""
Classical Object Detector: HOG + SVM.

Implements a traditional computer vision pipeline for object detection
using handcrafted features and a linear classifier. Serves as a baseline
for comparison against the YOLO deep-learning model.
"""

import cv2
import numpy as np
from sklearn.svm import LinearSVC
from src.utils.config import CLASSICAL
from src.preprocessing.image_pipeline import preprocess_for_classical


class HOGSVMDetector:
    """HOG + SVM detector wrapper."""

    def __init__(self, svm_path=None):
        self.hog = cv2.HOGDescriptor(
            CLASSICAL["hog_win_size"],
            CLASSICAL["hog_block_size"],
            CLASSICAL["hog_block_stride"],
            CLASSICAL["hog_cell_size"],
            CLASSICAL["hog_nbins"],
        )
        self.svm = None
        if svm_path:
            self.svm = LinearSVC(C=CLASSICAL["svm_c"])
            # Load coefficients from saved model if available
            # For now, fall back to default detector if load fails
            try:
                data = np.load(svm_path, allow_pickle=True).item()
                coefs = data["coef"]
                intercept = data.get("intercept", 0.0)
                self.hog.setSVMDetector(np.concatenate(([intercept], coefs.flatten())))
            except Exception:
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            # Use OpenCV's default people detector as zero-training baseline
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image: np.ndarray) -> list:
        """
        Detect objects using sliding window + HOG + SVM.

        Args:
            image: BGR image.

        Returns:
            List of detections, each as [class_id, confidence, x, y, w, h].
            For the default detector, class_id is always 0 (person).
        """
        preprocessed = preprocess_for_classical(image)

        # detectMultiScale returns (rectangles, weights)
        # rectangles: [[x, y, w, h], ...]
        # weights: confidence scores for each detection
        rects, weights = self.hog.detectMultiScale(
            preprocessed,
            winStride=(8, 8),
            padding=(4, 4),
            scale=CLASSICAL["scale_factor"],
        )

        detections = []
        for i, (x, y, w, h) in enumerate(rects):
            confidence = float(weights[i]) if i < len(weights) else 1.0
            detections.append([0, confidence, int(x), int(y), int(w), int(h)])

        return detections

    def train(self, positive_samples, negative_samples):
        """
        Train a custom linear SVM on HOG features.

        Args:
            positive_samples: List of positive image crops (64x128 grayscale).
            negative_samples: List of negative image crops (64x128 grayscale).
        """
        features = []
        labels = []

        win_size = CLASSICAL["hog_win_size"]

        for img in positive_samples:
            if img.shape[:2] != win_size[::-1]:
                img = cv2.resize(img, win_size)
            feat = self.hog.compute(img)
            features.append(feat.flatten())
            labels.append(1)

        for img in negative_samples:
            if img.shape[:2] != win_size[::-1]:
                img = cv2.resize(img, win_size)
            feat = self.hog.compute(img)
            features.append(feat.flatten())
            labels.append(0)

        X = np.array(features, dtype=np.float32)
        y = np.array(labels)

        self.svm = LinearSVC(C=CLASSICAL["svm_c"], max_iter=10000)
        self.svm.fit(X, y)

        # Convert sklearn coefs to OpenCV HOGDescriptor SVM format
        # OpenCV expects: [bias, coef_1, coef_2, ..., coef_n]
        coefs = self.svm.coef_.flatten().astype(np.float64)
        intercept = self.svm.intercept_[0]
        svm_detector = np.concatenate(([intercept], coefs))
        self.hog.setSVMDetector(svm_detector)


def main():
    import argparse
    from pathlib import Path
    import cv2

    parser = argparse.ArgumentParser(description="HOG+SVM Object Detector")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--svm", default=None, help="Path to saved SVM model (optional)")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    detector = HOGSVMDetector(svm_path=args.svm)
    detections = detector.detect(image)
    print(f"Detected {len(detections)} objects")

    from src.utils.visualization import draw_detections
    annotated = draw_detections(image, detections)
    out_path = output_dir / f"classical_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
