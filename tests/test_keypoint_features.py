"""
Unit tests for geometric feature extractor.
"""

import numpy as np
import pytest

from src.classical.feature_extractor import FeatureExtractor


def test_extractor_output_shape():
    """Feature vector should have consistent shape."""
    extractor = FeatureExtractor()
    # 21 landmarks with (x, y, z) each
    landmarks = [(0.1 * i, 0.1 * i, 0.0) for i in range(21)]
    features = extractor.extract(landmarks)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 1


def test_extractor_normalization():
    """Normalized features should have reasonable bounds."""
    extractor = FeatureExtractor()
    landmarks = [(0.5, 0.5, 0.0) for _ in range(21)]
    features = extractor.extract(landmarks)
    # After normalization, most values should be finite
    assert np.all(np.isfinite(features))
