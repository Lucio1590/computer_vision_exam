"""
Live webcam pipeline for dual-mode gesture recognition.

Supports:
    - CLASSICAL mode: MediaPipe keypoints + SVM
    - DEEP mode: YOLOv8 detection + CNN classification

Controls:
    - 'c' -> switch to CLASSICAL mode
    - 'd' -> switch to DEEP mode
    - 's' -> screenshot
    - 'q' -> quit
"""

import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.classical.hand_detector import HandDetector
from src.classical.feature_extractor import FeatureExtractor
from src.classical.gesture_classifier import GestureClassifier
from src.deep_learning.yolo_detector import YOLOHandDetector
from src.deep_learning.gesture_net import get_model
from src.config import (
    CLASSICAL,
    CNN,
    PREPROCESSING,
    GESTURE_CLASSES,
    SVM_KEYPOINTS_PATH,
    GESTURE_CNN_PATH,
    VISUALIZATION,
)


class GesturePipeline:
    """Live gesture recognition pipeline."""

    def __init__(self, mode="classical"):
        """
        Initialize pipeline.

        Args:
            mode: 'classical' or 'deep'.
        """
        self.mode = mode
        self.detector = None
        self.classifier = None
        self.yolo = None
        self.cnn = None
        self.cnn_transform = None
        self._load_models()

    def _load_models(self):
        """Load models based on current mode."""
        if self.mode == "classical":
            self.detector = HandDetector()
            self.extractor = FeatureExtractor()
            self.classifier = GestureClassifier(
                kernel=CLASSICAL["svm_kernel"],
                C=CLASSICAL["svm_c"],
            )
            if SVM_KEYPOINTS_PATH.exists():
                self.classifier.load(str(SVM_KEYPOINTS_PATH))
            else:
                print("WARNING: SVM model not found. Run train_svm.py first.")
        elif self.mode == "deep":
            self.yolo = YOLOHandDetector()
            self.cnn = get_model(num_classes=CNN["num_classes"])
            if GESTURE_CNN_PATH.exists():
                self.cnn.load_state_dict(
                    torch.load(GESTURE_CNN_PATH, map_location="cpu")
                )
            else:
                print("WARNING: CNN model not found. Run train_classifier.py first.")
            self.cnn.eval()

            size = PREPROCESSING["classifier_input_size"]
            mean = PREPROCESSING["normalize_mean"]
            std = PREPROCESSING["normalize_std"]
            self.cnn_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.

        Args:
            frame: BGR image from webcam.

        Returns:
            Annotated frame with detections/landmarks and labels.
        """
        annotated = frame.copy()
        start_time = time.perf_counter()

        if self.mode == "classical":
            annotated = self._process_classical(annotated)
        else:
            annotated = self._process_deep(annotated)

        # Draw FPS and mode
        fps = 1.0 / (time.perf_counter() - start_time + 1e-6)
        annotated = self._draw_hud(annotated, fps)

        return annotated

    def _process_classical(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with classical pipeline."""
        hands = self.detector.detect(frame)

        for landmarks in hands:
            frame = self.detector.draw_landmarks(frame, landmarks)

            if self.classifier.is_trained:
                features = self.extractor.extract(landmarks)
                pred, conf = self.classifier.predict(features)
                label = f"{GESTURE_CLASSES[pred]}: {conf:.2f}"

                # Draw label near wrist (landmark 0)
                h, w = frame.shape[:2]
                wx = int(landmarks[0][0] * w)
                wy = int(landmarks[0][1] * h)
                cv2.putText(
                    frame, label, (wx, wy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    VISUALIZATION["text_color"], 2
                )

        return frame

    def _process_deep(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with deep learning pipeline."""
        detections = self.yolo.detect(frame)

        for det in detections:
            cls_id, conf, x, y, w, h = det
            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)

            # Draw bbox
            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                VISUALIZATION["box_color"], 2
            )

            # Classify crop if CNN is loaded
            if self.cnn is not None and GESTURE_CNN_PATH.exists():
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    tensor = self.cnn_transform(crop).unsqueeze(0)
                    with torch.no_grad():
                        logits = self.cnn(tensor)
                        probs = F.softmax(logits, dim=1)
                        pred = int(torch.argmax(probs, dim=1)[0])
                        conf_cls = float(probs[0][pred])
                    label = f"{GESTURE_CLASSES[pred]}: {conf_cls:.2f}"
                else:
                    label = f"hand: {conf:.2f}"
            else:
                label = f"hand: {conf:.2f}"

            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                VISUALIZATION["text_color"], 2
            )

        return frame

    def _draw_hud(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS and mode indicator."""
        mode_text = f"MODE: {self.mode.upper()}"
        fps_text = f"FPS: {fps:.1f}"

        cv2.putText(
            frame, mode_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )
        cv2.putText(
            frame, fps_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )
        return frame

    def switch_mode(self, mode: str):
        """Switch between classical and deep mode."""
        if self.mode == mode:
            return

        # Release old models
        if self.detector:
            self.detector.close()
        self.detector = None
        self.classifier = None
        self.yolo = None
        self.cnn = None

        self.mode = mode
        self._load_models()
        print(f"Switched to {mode.upper()} mode")
