"""
YOLOv8 hand detector wrapper using Ultralytics.

Fine-tuned YOLOv8n for single-class hand detection.
"""

import numpy as np


class YOLOHandDetector:
    """Wrapper around Ultralytics YOLOv8 for hand detection."""

    def __init__(self, model_path: str = None):
        """
        Initialize YOLOv8 detector.

        Args:
            model_path: Path to fine-tuned YOLOv8 weights.
                        If None, uses pretrained YOLOv8n.
        """
        from ultralytics import YOLO
        from src.config import YOLO_HAND_MODEL_PATH, YOLO_HAND_MODEL_FALLBACK

        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        elif YOLO_HAND_MODEL_PATH.exists():
            self.model = YOLO(str(YOLO_HAND_MODEL_PATH))
        elif YOLO_HAND_MODEL_FALLBACK.exists():
            self.model = YOLO(str(YOLO_HAND_MODEL_FALLBACK))
        else:
            # Fallback to pretrained YOLOv8n (will detect 80 COCO classes)
            self.model = YOLO("yolov8n.pt")

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> list:
        """
        Detect hands in an image.

        Args:
            image: BGR image (H x W x 3).
            conf_threshold: Minimum confidence threshold.

        Returns:
            List of detections [class_id, confidence, x, y, w, h] in xywh format.
        """
        results = self.model(image, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls[0])
            x, y, w, h = box.xywh[0].tolist()
            detections.append([cls_id, conf, x, y, w, h])

        return detections

    def crop_hand(self, image: np.ndarray, bbox: list, padding: float = 0.1) -> np.ndarray:
        """
        Extract hand crop from image with optional padding.

        Args:
            image: Original BGR image.
            bbox: Bounding box [x, y, w, h] in absolute pixel coords.
            padding: Fraction of bbox size to add as padding.

        Returns:
            Cropped hand image.
        """
        import cv2

        h, w = image.shape[:2]
        x, y, bw, bh = bbox

        pad_x = int(bw * padding)
        pad_y = int(bh * padding)

        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(w, int(x + bw + pad_x))
        y2 = min(h, int(y + bh + pad_y))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image
        return crop


from pathlib import Path
