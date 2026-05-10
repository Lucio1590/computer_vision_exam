"""
Image preprocessing pipeline.

Encapsulates cv2.dnn.blobFromImage logic and classical preprocessing
in a modular, reusable interface.
"""

import cv2
import numpy as np
from src.utils.config import PREPROCESSING, CLASSICAL


def preprocess_for_yolo(image: np.ndarray) -> np.ndarray:
    """
    Transform an input image into a 4D blob suitable for YOLO inference.

    Args:
        image: Input image in BGR format (H x W x C).

    Returns:
        A 4D blob with shape (1, 3, 416, 416).
    """
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=PREPROCESSING["scalefactor"],
        size=PREPROCESSING["input_size"],
        mean=PREPROCESSING["mean"],
        swapRB=PREPROCESSING["swap_rb"],
        crop=PREPROCESSING["crop"],
    )
    return blob


def preprocess_for_classical(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for classical HOG+SVM detector.

    Steps:
        1. Convert to grayscale.
        2. Apply histogram equalization for illumination normalization.

    Args:
        image: Input image in BGR format.

    Returns:
        Grayscale, equalized image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


def resize_image(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize image to target dimensions.

    Args:
        image: Input image.
        target_size: (width, height).

    Returns:
        Resized image.
    """
    return cv2.resize(image, target_size)
