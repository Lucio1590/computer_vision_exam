#!/usr/bin/env python3
"""
Dataset download script.

Supports:
    - INRIA Person Dataset (for classical training)
    - COCO 2017 mini-validation subset (for evaluation)
"""

import argparse
import json
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANNOTATIONS_DIR = DATA_DIR / "annotations"


def download_inria():
    """Download INRIA Person dataset."""
    # TODO: Implement INRIA download and extraction
    # URL: http://pascal.inrialpes.fr/data/human/INRIAPerson.tar
    pass


def download_coco_mini(num_images=100):
    """
    Download a random subset of COCO 2017 validation images.

    Args:
        num_images: Number of images to sample (default 100).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotation_zip = DATA_DIR / "annotations_trainval2017.zip"

    if not annotation_zip.exists():
        print("Downloading COCO annotations...")
        urlretrieve(annotation_url, annotation_zip)
        with zipfile.ZipFile(annotation_zip, "r") as z:
            z.extractall(DATA_DIR)

    # Load instances_val2017.json
    ann_path = ANNOTATIONS_DIR / "instances_val2017.json"
    with open(ann_path, "r") as f:
        coco_data = json.load(f)

    images = coco_data["images"]
    sampled = random.sample(images, min(num_images, len(images)))
    image_ids = {img["id"] for img in sampled}

    # Filter annotations
    anns = [a for a in coco_data["annotations"] if a["image_id"] in image_ids]
    mini_coco = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": sampled,
        "annotations": anns,
        "categories": coco_data["categories"],
    }

    mini_ann_path = ANNOTATIONS_DIR / f"instances_mini{num_images}.json"
    with open(mini_ann_path, "w") as f:
        json.dump(mini_coco, f)
    print(f"Saved mini annotations to {mini_ann_path}")

    # Download images
    base_url = "http://images.cocodataset.org/"
    for img in tqdm(sampled, desc="Downloading images"):
        img_path = RAW_DIR / img["file_name"]
        if img_path.exists():
            continue
        url = f"{base_url}val2017/{img['file_name']}"
        try:
            urlretrieve(url, img_path)
        except Exception as e:
            print(f"Failed to download {img['file_name']}: {e}")

    print(f"Downloaded {len(sampled)} images to {RAW_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--inria", action="store_true", help="Download INRIA Person dataset"
    )
    parser.add_argument(
        "--mini", type=int, default=0, help="Download N random COCO val images"
    )
    args = parser.parse_args()

    if args.inria:
        download_inria()
    if args.mini > 0:
        download_coco_mini(args.mini)

    if not args.inria and args.mini <= 0:
        print("No action specified. Use --inria or --mini N")


if __name__ == "__main__":
    main()
