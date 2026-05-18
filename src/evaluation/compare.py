"""
Side-by-side comparison of Classical vs Deep Learning detectors.

Runs both detectors on the same images, collects metrics,
and generates a comparison report.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from src.evaluation.benchmark import benchmark_detector
from src.evaluation.metrics import compute_map
from src.utils.visualization import draw_detections


def run_comparison(classical_detector, yolo_detector, test_images, output_dir):
    """
    Run both detectors on test images and save comparison results.

    Args:
        classical_detector: Instance with detect(image) method.
        yolo_detector: Instance with detect(image) method.
        test_images: List of image paths or loaded images.
        output_dir: Directory to save visualizations and report.

    Returns:
        Comparison dictionary with metrics and file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for item in test_images:
        if isinstance(item, (str, Path)):
            img = cv2.imread(str(item))
            if img is not None:
                images.append(img)
        else:
            images.append(item)

    if len(images) == 0:
        raise ValueError("No valid images provided for comparison.")

    # Benchmark
    print("Benchmarking classical detector...")
    classical_bench = benchmark_detector(classical_detector, images)

    print("Benchmarking YOLO detector...")
    yolo_bench = benchmark_detector(yolo_detector, images)

    # Run detection and save visualizations
    classical_results = []
    yolo_results = []

    for i, image in enumerate(images):
        c_dets = classical_detector.detect(image)
        y_dets = yolo_detector.detect(image)

        classical_results.append({
            "image_id": i,
            "boxes": [[d[2], d[3], d[4], d[5]] for d in c_dets],
            "scores": [d[1] for d in c_dets],
            "class_ids": [d[0] for d in c_dets],
        })

        yolo_results.append({
            "image_id": i,
            "boxes": [[d[2], d[3], d[4], d[5]] for d in y_dets],
            "scores": [d[1] for d in y_dets],
            "class_ids": [d[0] for d in y_dets],
        })

        # Save annotated images
        c_vis = draw_detections(image.copy(), c_dets)
        y_vis = draw_detections(image.copy(), y_dets)

        cv2.imwrite(str(output_dir / f"classical_{i}.jpg"), c_vis)
        cv2.imwrite(str(output_dir / f"yolo_{i}.jpg"), y_vis)

    report = {
        "num_images": len(images),
        "classical": {
            "fps": classical_bench["mean_fps"],
            "latency_ms": classical_bench["mean_latency_ms"],
            "total_detections": sum(len(r["boxes"]) for r in classical_results),
        },
        "yolo": {
            "fps": yolo_bench["mean_fps"],
            "latency_ms": yolo_bench["mean_latency_ms"],
            "total_detections": sum(len(r["boxes"]) for r in yolo_results),
        },
    }

    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Comparison complete. Results saved to {output_dir}")
    return report


def main():
    import argparse
    from pathlib import Path
    import cv2
    from src.classical.detector import HOGSVMDetector
    from src.inference.yolo_detector import YOLODetector

    parser = argparse.ArgumentParser(description="Compare Classical vs YOLO detectors")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to test images")
    parser.add_argument("--output", default="docs/results", help="Output directory")
    args = parser.parse_args()

    image_paths = [Path(p) for p in args.images]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    classical = HOGSVMDetector()
    yolo = YOLODetector()

    report = run_comparison(classical, yolo, image_paths, output_dir)
    print("\n--- Comparison Report ---")
    print(f"Images processed: {report['num_images']}")
    print(f"Classical FPS: {report['classical']['fps']:.2f}")
    print(f"YOLO FPS: {report['yolo']['fps']:.2f}")


if __name__ == "__main__":
    main()
