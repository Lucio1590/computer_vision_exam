"""
MediaPipe Hands wrapper for hand landmark detection.

Uses MediaPipe Tasks API (HandLandmarker) to detect up to 2 hands per frame
and returns 21 normalized landmarks for each detected hand.
"""

import cv2
import numpy as np
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

from src.config import CLASSICAL, MODELS_DIR

# Default model path
DEFAULT_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"


class HandDetector:
    """Wrapper around MediaPipe HandLandmarker for gesture recognition."""

    def __init__(self, model_path=None, num_hands=None,
                 min_detection_confidence=None):
        """
        Initialize MediaPipe HandLandmarker.

        Args:
            model_path: Path to hand_landmarker.task model file.
            num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum confidence for hand detection.
        """
        model_path = str(model_path or DEFAULT_LANDMARKER_MODEL)
        if not DEFAULT_LANDMARKER_MODEL.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=num_hands or CLASSICAL["max_num_hands"],
            min_hand_detection_confidence=
            min_detection_confidence or CLASSICAL["min_detection_confidence"],
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, image: np.ndarray) -> list:
        """
        Detect hands in an image.

        Args:
            image: BGR image (H x W x 3).

        Returns:
            List of hand landmarks, each as list of 21 (x, y, z) tuples
            in normalized coordinates [0, 1].
        """
        from mediapipe import Image as MPImage, ImageFormat

        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)

        results = self.landmarker.detect(mp_image)

        hands_landmarks = []
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.append((lm.x, lm.y, lm.z))
                hands_landmarks.append(landmarks)

        return hands_landmarks

    def draw_landmarks(self, image: np.ndarray, landmarks: list) -> np.ndarray:
        """
        Draw hand landmarks and skeleton on the image.

        Args:
            image: BGR image.
            landmarks: List of 21 (x, y, z) normalized landmarks.

        Returns:
            Annotated image.
        """
        h, w, _ = image.shape
        annotated = image.copy()

        # MediaPipe hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]

        for start_idx, end_idx in connections:
            cv2.line(annotated, pts[start_idx], pts[end_idx], (255, 0, 0), 2)

        for pt in pts:
            cv2.circle(annotated, pt, 4, (0, 255, 0), -1)

        return annotated

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
