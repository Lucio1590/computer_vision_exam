"""
Build keypoint dataset from HaGRID using MediaPipe Hands.

Runs MediaPipe on each image, extracts 21 landmarks, computes geometric features,
and saves CSV files for SVM training.

Output:
    data/processed/keypoints/keypoints_train.csv
    data/processed/keypoints/keypoints_val.csv
    data/processed/keypoints/keypoints_test.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import GESTURE_CLASSES, KEYPOINTS_DATASET_DIR
from src.classical.hand_detector import HandDetector
from src.classical.feature_extractor import FeatureExtractor

random.seed(42)


def build_keypoint_dataset(raw_dir: Path, output_dir: Path, target_classes: list):
    """
    Extract MediaPipe keypoints and geometric features from HaGRID images.

    Args:
        raw_dir: Path to extracted HaGRID sample dataset.
        output_dir: Path to save CSV files.
        target_classes: List of gesture classes to include.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = raw_dir / "hagrid-sample-30k-384p"
    images_root = dataset_root / "hagrid_30k"
    ann_root = dataset_root / "ann_train_val"

    detector = HandDetector()
    extractor = FeatureExtractor()

    # Collect all samples per class
    class_samples = {cls: [] for cls in target_classes}

    for cls in target_classes:
        ann_path = ann_root / f"{cls}.json"
        if not ann_path.exists():
            continue

        with open(ann_path, "r") as f:
            annotations = json.load(f)

        img_dir = images_root / f"train_val_{cls}"
        if not img_dir.exists():
            continue

        for img_uuid, ann in annotations.items():
            img_path = img_dir / f"{img_uuid}.jpg"
            if not img_path.exists():
                continue

            user_id = ann.get("user_id", "unknown")
            class_samples[cls].append({
                "uuid": img_uuid,
                "img_path": img_path,
                "label": cls,
                "user_id": user_id,
            })

    # Stratified split by user_id
    for cls in target_classes:
        samples = class_samples[cls]
        if not samples:
            continue

        users = {}
        for s in samples:
            users.setdefault(s["user_id"], []).append(s)

        user_ids = list(users.keys())
        random.shuffle(user_ids)

        n = len(user_ids)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)

        train_users = set(user_ids[:n_train])
        val_users = set(user_ids[n_train:n_train + n_val])
        test_users = set(user_ids[n_train + n_val:])

        class_samples[cls] = {
            "train": [s for uid in train_users for s in users[uid]],
            "val": [s for uid in val_users for s in users[uid]],
            "test": [s for uid in test_users for s in users[uid]],
        }

    # Write CSVs per split
    splits = ["train", "val", "test"]
    for split in splits:
        csv_path = output_dir / f"keypoints_{split}.csv"
        with open(csv_path, "w", newline="") as csvfile:
            # Header: features + label
            # We don't know feature dim yet; write dynamically
            writer = None
            total = sum(len(class_samples[cls][split]) for cls in target_classes)

            pbar = tqdm(total=total, desc=f"Keypoints/{split}")
            for cls in target_classes:
                for sample in class_samples[cls][split]:
                    img = cv2.imread(str(sample["img_path"]))
                    if img is None:
                        pbar.update(1)
                        continue

                    hands = detector.detect(img)
                    if not hands:
                        pbar.update(1)
                        continue

                    # Use the first detected hand
                    features = extractor.extract(hands[0])

                    if writer is None:
                        # Initialize writer with header
                        fieldnames = [f"f{i}" for i in range(len(features))] + ["label"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()

                    row = {f"f{i}": v for i, v in enumerate(features)}
                    row["label"] = sample["label"]
                    writer.writerow(row)
                    pbar.update(1)
            pbar.close()

    detector.close()
    print(f"\nKeypoint dataset built at: {output_dir}")
    for split in splits:
        csv_path = output_dir / f"keypoints_{split}.csv"
        if csv_path.exists():
            # Count lines
            with open(csv_path) as f:
                n = sum(1 for _ in f) - 1  # minus header
            print(f"  {split}: {n} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Build keypoint dataset from HaGRID"
    )
    parser.add_argument("--raw", default="data/raw/hagrid_subset",
                        help="Path to extracted HaGRID dataset")
    parser.add_argument("--output", default=str(KEYPOINTS_DATASET_DIR),
                        help="Output keypoints directory")
    parser.add_argument("--classes", nargs="+", default=GESTURE_CLASSES,
                        help="Target gesture classes")
    args = parser.parse_args()

    build_keypoint_dataset(Path(args.raw), Path(args.output), args.classes)


if __name__ == "__main__":
    main()
