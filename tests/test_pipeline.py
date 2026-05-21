"""
End-to-end pipeline smoke tests for Hand Gesture Recognition.

Tests that all modules import correctly and basic operations
work on synthetic data.
"""

import numpy as np
import pytest

from src.preprocessing.image_pipeline import preprocess_for_yolo, preprocess_for_classical


class TestPreprocessing:
    def test_yolo_blob_shape(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        blob = preprocess_for_yolo(img)
        assert blob.shape == (1, 3, 416, 416)

    def test_classical_grayscale(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = preprocess_for_classical(img)
        assert len(out.shape) == 2
        assert out.shape == (480, 640)


class TestHandDetector:
    def test_import(self):
        from src.classical.hand_detector import HandDetector
        assert HandDetector is not None

    def test_instantiation(self):
        from src.classical.hand_detector import HandDetector
        det = HandDetector()
        assert det is not None


class TestFeatureExtractor:
    def test_extractor_shape(self):
        from src.classical.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        landmarks = [(0.1 * i, 0.1 * i, 0.0) for i in range(21)]
        features = extractor.extract(landmarks)
        assert isinstance(features, np.ndarray)


class TestGestureClassifier:
    def test_import(self):
        from src.classical.gesture_classifier import GestureClassifier
        assert GestureClassifier is not None


class TestYOLOHandDetector:
    def test_import(self):
        from src.deep_learning.yolo_detector import YOLOHandDetector
        assert YOLOHandDetector is not None


class TestGestureNet:
    def test_import(self):
        torch = pytest.importorskip("torch")
        from src.deep_learning.gesture_net import GestureNet
        assert GestureNet is not None


class TestWebcamStreamer:
    def test_import(self):
        from src.webcam.streamer import WebcamStreamer
        assert WebcamStreamer is not None


class TestGesturePipeline:
    def test_import(self):
        from src.webcam.pipeline import GesturePipeline
        assert GesturePipeline is not None

    def test_mode_switch(self):
        from src.webcam.pipeline import GesturePipeline
        pipeline = GesturePipeline(mode="classical")
        assert pipeline.mode == "classical"
        pipeline.switch_mode("deep")
        assert pipeline.mode == "deep"
