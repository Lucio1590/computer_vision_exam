"""
SVM-based gesture classifier.

Wraps scikit-learn SVM for classifying hand gestures from geometric features.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


class GestureClassifier:
    """SVM gesture classifier wrapper."""

    def __init__(self, kernel="rbf", C=1.0):
        """
        Initialize SVM classifier.

        Args:
            kernel: SVM kernel type ('rbf', 'linear', 'poly').
            C: Regularization parameter.
        """
        self.scaler = StandardScaler()
        self.svm = SVC(kernel=kernel, C=C, probability=True)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the SVM on feature vectors.

        Args:
            X: Feature matrix (N_samples, N_features).
            y: Labels (N_samples,).
        """
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, features: np.ndarray) -> tuple:
        """
        Predict gesture class and confidence.

        Args:
            features: Feature vector or matrix.

        Returns:
            Tuple of (predicted_class_index, confidence).
        """
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained yet.")

        features = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(features)
        pred = self.svm.predict(X_scaled)[0]
        proba = self.svm.predict_proba(X_scaled)[0]
        conf = float(np.max(proba))

        # Convert string label to integer index if needed
        if isinstance(pred, str):
            from src.config import GESTURE_CLASSES
            pred = GESTURE_CLASSES.index(pred)

        return pred, conf

    def save(self, path: str):
        """Save model and scaler to disk."""
        joblib.dump({"svm": self.svm, "scaler": self.scaler}, path)

    def load(self, path: str):
        """Load model and scaler from disk."""
        data = joblib.load(path)
        self.svm = data["svm"]
        self.scaler = data["scaler"]
        self.is_trained = True
