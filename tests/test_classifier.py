"""
Unit tests for gesture classifiers (SVM and CNN).
"""

import numpy as np
import pytest


def test_svm_predicts_valid_class():
    """SVM should predict one of the 6 gesture classes."""
    from src.classical.gesture_classifier import GestureClassifier
    from src.config import NUM_GESTURE_CLASSES

    clf = GestureClassifier()
    # Mock training with random data
    X = np.random.rand(20, 10)
    y = np.random.randint(0, NUM_GESTURE_CLASSES, 20)
    clf.train(X, y)

    pred, conf = clf.predict(X[0])
    assert 0 <= pred < NUM_GESTURE_CLASSES
    assert 0 <= conf <= 1


def test_cnn_output_shape():
    """CNN should output tensor of shape [B, num_classes]."""
    torch = pytest.importorskip("torch")
    from src.deep_learning.gesture_net import GestureNet
    from src.config import NUM_GESTURE_CLASSES

    model = GestureNet(num_classes=NUM_GESTURE_CLASSES)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, NUM_GESTURE_CLASSES)


def test_cnn_probabilities_sum_to_one():
    """After softmax, probabilities should sum to ~1."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from src.deep_learning.gesture_net import GestureNet
    from src.config import NUM_GESTURE_CLASSES

    model = GestureNet(num_classes=NUM_GESTURE_CLASSES)
    x = torch.randn(1, 3, 224, 224)
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
