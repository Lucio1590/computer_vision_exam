#!/usr/bin/env python3
"""
Automated download script for YOLO model files.

Downloads YOLOv3 weights, config, and COCO class names into models/.
Run this first before attempting inference.
"""

import requests
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048,  # ~236 MB
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": None,
    },
    "coco.names": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "size": None,
    },
}


def download_file(name: str, url: str, dest: Path, expected_size: int = None):
    """Download a file with progress bar."""
    if dest.exists():
        print(f"[SKIP] {name} already exists at {dest}")
        return

    print(f"[DOWNLOAD] {name} from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = expected_size or int(response.headers.get("content-length", 0))
    block_size = 8192

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"[DONE] {name} saved to {dest}")


def main():
    for name, info in FILES.items():
        dest = MODELS_DIR / name
        download_file(name, info["url"], dest, info.get("size"))
    print("\nAll model files are ready. You can now run inference.")


if __name__ == "__main__":
    main()
