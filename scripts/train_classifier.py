"""
Entrypoint for training the PyTorch gesture classifier.

Wraps src.deep_learning.train_classifier with default paths from config.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path

from src.config import GESTURE_CNN_PATH, CLASSIFIER_DATASET_DIR
from src.deep_learning.train_classifier import train_classifier


def main():
    parser = argparse.ArgumentParser(description="Train gesture CNN classifier")
    parser.add_argument("--data", default=str(CLASSIFIER_DATASET_DIR))
    parser.add_argument("--output", default=str(GESTURE_CNN_PATH))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    train_classifier(Path(args.data), Path(args.output), args.epochs)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
