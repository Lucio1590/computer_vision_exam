"""
End-to-end pipeline smoke tests.
"""

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from src.classical.detector import HOGSVMDetector
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


class TestClassicalDetector:
    def test_instantiation(self):
        det = HOGSVMDetector()
        assert det is not None

    def test_detect_returns_list(self):
        det = HOGSVMDetector()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = det.detect(img)
        assert isinstance(dets, list)
        # Each detection should be [class_id, confidence, x, y, w, h]
        for d in dets:
            assert len(d) == 6


class TestYOLODetector:
    def test_instantiation(self):
        # This requires model files to be present
        try:
            from src.inference.yolo_detector import YOLODetector
            det = YOLODetector()
            assert det is not None
        except FileNotFoundError:
            if pytest:
                pytest.skip("YOLO model files not downloaded")
            else:
                print("SKIP: YOLO model files not downloaded")

    def test_detect_requires_weights(self):
        try:
            from src.inference.yolo_detector import YOLODetector
            det = YOLODetector()
            img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
            dets = det.detect(img)
            assert isinstance(dets, list)
        except FileNotFoundError:
            if pytest:
                pytest.skip("YOLO model files not downloaded")
            else:
                print("SKIP: YOLO model files not downloaded")
