"""
Entrypoint for training the classical SVM gesture classifier.

Wraps src.classical.train with default paths from config.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path

from src.config import SVM_KEYPOINTS_PATH, KEYPOINTS_DATASET_DIR
from src.classical.train import train_svm


def main():
    parser = argparse.ArgumentParser(description="Train SVM gesture classifier")
    parser.add_argument("--train", default=str(KEYPOINTS_DATASET_DIR / "keypoints_train.csv"))
    parser.add_argument("--val", default=str(KEYPOINTS_DATASET_DIR / "keypoints_val.csv"))
    parser.add_argument("--output", default=str(SVM_KEYPOINTS_PATH))
    args = parser.parse_args()

    train_svm(Path(args.train), Path(args.val), Path(args.output))
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
