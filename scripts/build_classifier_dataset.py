"""
Build classifier dataset from HaGRID by extracting hand crops.

Uses ground-truth bounding boxes to crop and resize hand regions to 224x224.
Output structure:
    data/processed/classifier/{train,val,test}/{class_name}/{uuid}.jpg

Split ratios: 70% train, 15% val, 15% test (by subject/user_id when possible).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import CLASSIFIER_DATASET_DIR, GESTURE_CLASSES, PREPROCESSING

random.seed(42)


def build_classifier_dataset(raw_dir: Path, output_dir: Path, target_classes: list):
    """
    Extract hand crops for CNN classifier training.

    Args:
        raw_dir: Path to extracted HaGRID sample dataset.
        output_dir: Path to save cropped images.
        target_classes: List of gesture classes to include.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate image and annotation directories
    dataset_root = raw_dir / "hagrid-sample-30k-384p"
    images_root = dataset_root / "hagrid_30k"
    ann_root = dataset_root / "ann_train_val"

    # Collect all samples per class with user_id for stratified split
    class_samples = {cls: [] for cls in target_classes}

    for cls in target_classes:
        ann_path = ann_root / f"{cls}.json"
        if not ann_path.exists():
            print(f"WARNING: Annotation file not found: {ann_path}")
            continue

        with open(ann_path, "r") as f:
            annotations = json.load(f)

        img_dir = images_root / f"train_val_{cls}"
        if not img_dir.exists():
            print(f"WARNING: Image directory not found: {img_dir}")
            continue

        for img_uuid, ann in annotations.items():
            img_path = img_dir / f"{img_uuid}.jpg"
            if not img_path.exists():
                continue

            bboxes = ann.get("bboxes", [])
            labels = ann.get("labels", [])
            user_id = ann.get("user_id", "unknown")

            # Find bbox matching the target class
            for bbox, label in zip(bboxes, labels):
                if label == cls:
                    class_samples[cls].append({
                        "uuid": img_uuid,
                        "img_path": img_path,
                        "bbox": bbox,
                        "user_id": user_id,
                    })
                    break  # Only one crop per image per class

    # Stratified split by user_id to avoid data leakage
    for cls in target_classes:
        samples = class_samples[cls]
        if not samples:
            continue

        # Group by user_id
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

        splits = {
            "train": [s for uid in train_users for s in users[uid]],
            "val": [s for uid in val_users for s in users[uid]],
            "test": [s for uid in test_users for s in users[uid]],
        }

        for split_name, split_samples in splits.items():
            split_dir = output_dir / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)

            for sample in tqdm(split_samples, desc=f"{cls}/{split_name}", leave=False):
                img = cv2.imread(str(sample["img_path"]))
                if img is None:
                    continue

                h, w = img.shape[:2]
                x, y, bw, bh = sample["bbox"]

                # Convert normalized bbox to pixel coords
                px = int(x * w)
                py = int(y * h)
                pw = int(bw * w)
                ph = int(bh * h)

                # Add 10% padding
                pad_x = int(pw * 0.1)
                pad_y = int(ph * 0.1)

                x1 = max(0, px - pad_x)
                y1 = max(0, py - pad_y)
                x2 = min(w, px + pw + pad_x)
                y2 = min(h, py + ph + pad_y)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Resize to classifier input size
                size = PREPROCESSING["classifier_input_size"]
                crop_resized = cv2.resize(crop, (size, size))

                out_path = split_dir / f"{sample['uuid']}.jpg"
                cv2.imwrite(str(out_path), crop_resized)

    print(f"\nClassifier dataset built at: {output_dir}")
    # Print stats
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        if split_dir.exists():
            total = sum(1 for _ in split_dir.rglob("*.jpg"))
            print(f"  {split}: {total} images")


def main():
    parser = argparse.ArgumentParser(
        description="Build classifier dataset from HaGRID"
    )
    parser.add_argument("--raw", default="data/raw/hagrid_subset",
                        help="Path to extracted HaGRID dataset")
    parser.add_argument("--output", default=str(CLASSIFIER_DATASET_DIR),
                        help="Output classifier dataset directory")
    parser.add_argument("--classes", nargs="+", default=GESTURE_CLASSES,
                        help="Target gesture classes")
    args = parser.parse_args()

    build_classifier_dataset(Path(args.raw), Path(args.output), args.classes)


if __name__ == "__main__":
    main()
