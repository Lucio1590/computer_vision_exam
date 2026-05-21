"""
Download HaGRID subset for hand gesture recognition.

Supports downloading via kagglehub or kaggle API.
Filters only the 6 target classes: like, dislike, ok, palm, fist, peace.

Usage:
    python scripts/download_hagrid.py --output data/raw/

Environment:
    Set KAGGLE_USERNAME and KAGGLE_KEY as environment variables,
    or place kaggle.json in ~/.kaggle/ for API authentication.
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

# HaGRID dataset identifier on Kaggle
# Using the 30k sample (384p) which includes images + annotations
KAGGLE_DATASET = "innominate817/hagrid-sample-30k-384p"

# Target gesture classes
TARGET_CLASSES = ["like", "dislike", "ok", "palm", "fist", "peace"]


def download_with_kagglehub(output_dir: Path):
    """
    Download HaGRID using kagglehub (recommended).

    Args:
        output_dir: Directory to extract dataset.

    Returns:
        Path to downloaded dataset directory.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub not installed. Run: pip install kagglehub"
        )

    print(f"Downloading HaGRID via kagglehub...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    return Path(path)


def download_with_kaggle_api(output_dir: Path):
    """
    Download HaGRID using Kaggle API CLI.

    Args:
        output_dir: Directory to extract dataset.

    Returns:
        Path to downloaded dataset directory.
    """
    import subprocess

    print(f"Downloading HaGRID via Kaggle API...")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(output_dir),
        "--unzip"
    ]
    subprocess.run(cmd, check=True)
    return output_dir


def filter_classes(source_dir: Path, target_dir: Path, classes: list):
    """
    Copy only target class folders from downloaded dataset.

    The sample dataset may be organized differently (flat images + CSV/JSON
    annotations or class folders). This function handles both layouts.

    Args:
        source_dir: Root of extracted HaGRID dataset.
        target_dir: Output directory for filtered subset.
        classes: List of class names to keep.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Try class-folder layout first
    found_any = False
    for cls in classes:
        src = source_dir / cls
        dst = target_dir / cls
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {cls}")
            found_any = True

    if not found_any:
        print("  WARNING: No class folders found. Dataset may use a different layout.")
        print("  Copying entire dataset for manual inspection.")
        for item in source_dir.iterdir():
            dst = target_dir / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            elif item.is_file():
                shutil.copy2(item, dst)

    print(f"  Subset saved to {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download HaGRID subset for hand gesture recognition"
    )
    parser.add_argument(
        "--output", default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--classes", nargs="+", default=TARGET_CLASSES,
        help="List of gesture classes to download"
    )
    parser.add_argument(
        "--method", choices=["kagglehub", "kaggle_api"], default="kagglehub",
        help="Download method"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    if args.method == "kagglehub":
        downloaded_path = download_with_kagglehub(output_dir)
    else:
        downloaded_path = download_with_kaggle_api(output_dir)

    print(f"Downloaded to: {downloaded_path}")

    # Filter classes
    subset_dir = output_dir / "hagrid_subset"
    print(f"\nFiltering classes: {args.classes}")
    filter_classes(downloaded_path, subset_dir, args.classes)

    print(f"\nDone. Dataset saved to: {subset_dir}")
    print(f"Explore the structure with: ls -R {subset_dir}")
    print(f"Next steps:")
    print(f"  python scripts/build_yolo_dataset.py")
    print(f"  python scripts/build_classifier_dataset.py")
    print(f"  python scripts/build_keypoint_dataset.py")


if __name__ == "__main__":
    main()
