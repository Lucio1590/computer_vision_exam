"""
Entrypoint for training YOLOv8n hand detector.

Wraps src.deep_learning.train_yolo with default paths from config.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path

from src.config import YOLO_HAND_MODEL_PATH, YOLO_DATASET_DIR
from src.deep_learning.train_yolo import train_yolo


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n hand detector")
    parser.add_argument("--data", default=str(YOLO_DATASET_DIR / "data.yaml"))
    parser.add_argument("--output", default=str(YOLO_HAND_MODEL_PATH.parent))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    train_yolo(Path(args.data), Path(args.output), args.epochs)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
