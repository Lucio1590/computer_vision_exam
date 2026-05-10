"""
Benchmarking utilities for measuring detector performance.

Provides FPS, latency, and throughput measurements.
"""

import time
import numpy as np


def benchmark_detector(detector, images, warmup=3):
    """
    Benchmark a detector's inference speed.

    Args:
        detector: Object with a detect(image) method.
        images: List of images to run inference on.
        warmup: Number of warmup iterations before timing.

    Returns:
        Dictionary with mean_fps, std_fps, total_time, num_images.
    """
    # Warmup
    if len(images) > 0:
        for _ in range(warmup):
            detector.detect(images[0])

    # Timing
    times = []
    for image in images:
        start = time.perf_counter()
        detector.detect(image)
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    fps = 1.0 / (times + 1e-10)

    return {
        "mean_fps": float(np.mean(fps)),
        "std_fps": float(np.std(fps)),
        "mean_latency_ms": float(np.mean(times) * 1000),
        "total_time": float(np.sum(times)),
        "num_images": len(images),
    }
