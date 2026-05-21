"""
Convert HaGRID annotations to YOLO format for hand detection.

Creates single-class dataset (class 0 = hand) with normalized bbox labels.
Output structure:
    data/processed/yolo/{train,val,test}/images
    data/processed/yolo/{train,val,test}/labels

All hand bboxes (any gesture) are labeled as class 0.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm

from src.config import GESTURE_CLASSES, YOLO_DATASET_DIR

random.seed(42)


def build_yolo_dataset(raw_dir: Path, output_dir: Path, target_classes: list):
    """
    Build YOLO-formatted dataset from HaGRID raw data.

    Args:
        raw_dir: Path to extracted HaGRID sample dataset.
        output_dir: Path to save YOLO dataset.
        target_classes: List of gesture classes to include.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = raw_dir / "hagrid-sample-30k-384p"
    images_root = dataset_root / "hagrid_30k"
    ann_root = dataset_root / "ann_train_val"

    # Collect all images with their bboxes
    all_samples = []

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

            bboxes = ann.get("bboxes", [])
            user_id = ann.get("user_id", "unknown")

            # All bboxes in this image are hands (class 0)
            if bboxes:
                all_samples.append({
                    "uuid": img_uuid,
                    "img_path": img_path,
                    "bboxes": bboxes,
                    "user_id": user_id,
                })

    # Stratified split by user_id
    users = {}
    for s in all_samples:
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

    # Create YOLO dataset structure
    for split_name, split_samples in splits.items():
        img_out_dir = output_dir / split_name / "images"
        lbl_out_dir = output_dir / split_name / "labels"
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(split_samples, desc=f"YOLO/{split_name}", leave=False):
            # Copy image
            img_dest = img_out_dir / f"{sample['uuid']}.jpg"
            shutil.copy2(sample["img_path"], img_dest)

            # Write YOLO label file
            label_dest = lbl_out_dir / f"{sample['uuid']}.txt"
            with open(label_dest, "w") as f:
                for bbox in sample["bboxes"]:
                    # HaGRID bbox: [top_left_x, top_left_y, width, height] (normalized)
                    # YOLO bbox: [class_id, center_x, center_y, width, height] (normalized)
                    x, y, w, h = bbox
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # Write dataset YAML config for Ultralytics
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n")
        f.write("\n")
        f.write("nc: 1\n")
        f.write("names: ['hand']\n")

    print(f"\nYOLO dataset built at: {output_dir}")
    for split in ["train", "val", "test"]:
        n = len(list((output_dir / split / "images").glob("*.jpg")))
        print(f"  {split}: {n} images")
    print(f"  YAML config: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Build YOLO dataset from HaGRID")
    parser.add_argument("--raw", default="data/raw/hagrid_subset",
                        help="Path to extracted HaGRID dataset")
    parser.add_argument("--output", default=str(YOLO_DATASET_DIR),
                        help="Output YOLO dataset directory")
    parser.add_argument("--classes", nargs="+", default=GESTURE_CLASSES,
                        help="Target gesture classes")
    args = parser.parse_args()

    build_yolo_dataset(Path(args.raw), Path(args.output), args.classes)


if __name__ == "__main__":
    main()
