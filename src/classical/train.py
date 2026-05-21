"""
Training pipeline for the classical MediaPipe + SVM gesture classifier.

Loads keypoint CSVs, trains StandardScaler + SVM, and saves the model.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.classical.gesture_classifier import GestureClassifier
from src.config import GESTURE_CLASSES


def train_svm(train_csv: Path, val_csv: Path, output_path: Path):
    """
    Train SVM on extracted keypoint features.

    Args:
        train_csv: Path to training features CSV.
        val_csv: Path to validation features CSV.
        output_path: Path to save trained model.
    """
    print(f"Loading training data from {train_csv}")
    train_df = pd.read_csv(train_csv)
    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values

    print(f"Loading validation data from {val_csv}")
    val_df = pd.read_csv(val_csv)
    X_val = val_df.drop(columns=["label"]).values
    y_val = val_df["label"].values

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    clf = GestureClassifier()
    clf.train(X_train, y_train)

    # Validate
    val_preds = []
    for x in X_val:
        pred, _ = clf.predict(x)
        val_preds.append(pred)

    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {val_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, val_preds, labels=GESTURE_CLASSES))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(output_path))
    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SVM gesture classifier")
    parser.add_argument("--train", required=True, help="Path to train CSV")
    parser.add_argument("--val", required=True, help="Path to val CSV")
    parser.add_argument("--output", default="models/svm_keypoints.pkl",
                        help="Output model path")
    args = parser.parse_args()

    train_svm(Path(args.train), Path(args.val), Path(args.output))


if __name__ == "__main__":
    main()
