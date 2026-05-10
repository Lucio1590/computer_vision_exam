"""
Classical Object Detector: HOG + SVM.

Implements a traditional computer vision pipeline for object detection
using handcrafted features and a linear classifier. Serves as a baseline
for comparison against the YOLO deep-learning model.
"""

import cv2
import numpy as np
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
            # TODO: Load custom trained SVM
            pass
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
        # TODO: Implement detection logic
        # 1. Preprocess image (grayscale + equalization)
        # 2. Run HOG detectMultiScale:
        #    found, weights = self.hog.detectMultiScale(preprocessed, ...)
        # 3. Convert outputs to [class_id, confidence, x, y, w, h] format
        # 4. Apply custom NMS if needed (HOG already groups rectangles)
        pass

    def train(self, positive_samples, negative_samples):
        """
        Train a custom linear SVM on HOG features.

        Args:
            positive_samples: List of positive image crops (64x128).
            negative_samples: List of negative image crops.
        """
        # TODO: Extract HOG features from all samples
        # TODO: Train sklearn.svm.LinearSVC
        # TODO: Set trained SVM to HOG descriptor
        pass
