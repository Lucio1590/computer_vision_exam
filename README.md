# Real-Time Urban Surveillance & Object Detection Framework

A modular, high-performance object detection system comparing a classical HOG+SVM baseline against a deep-learning YOLO detector. Built with OpenCV DNN for real-time traffic monitoring scenarios.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.5+-green.svg)](https://opencv.org/)

---

## Overview

This project demonstrates a complete computer vision pipeline:

| Stage | Classical (HOG+SVM) | Deep Learning (YOLOv3/v4) |
|-------|---------------------|---------------------------|
| Preprocessing | Grayscale, histogram equalization | Blob creation, normalization, BGR to RGB |
| Features | HOG handcrafted descriptors | Learned CNN features (Darknet-53) |
| Detection | Sliding window + linear SVM | Single-shot regression |
| Post-processing | Custom IoU + manual NMS | Confidence filter + OpenCV NMSBoxes |
| Evaluation | Precision, Recall, F1 | mAP, IoU, FPS |

### Design Principles
- **Modularity** — Swap backbones without touching visualization or metrics logic
- **Explainability** — Custom IoU and NMS implementations with mathematical transparency
- **Clean Code** — DRY configuration handler, KISS visualization, consistent bounding box format `[class_id, confidence, x, y, w, h]`

---

## Quick Start

### 1. Setup Environment

```bash
# Option A: Conda
conda env create -f environment.yml
conda activate cv-exam

# Option B: venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Models

```bash
python scripts/download_models.py
```

Downloads:
- `models/yolov3.weights` (~236 MB)
- `models/yolov3.cfg`
- `models/coco.names`

### 3. Download Test Data (Optional)

```bash
# Lightweight: 100 random COCO val images (~150 MB)
python scripts/download_data.py --mini 100

# Or place your own images/videos in data/raw/
```

### 4. Run Inference

```bash
# YOLO detection
python -m src.inference.yolo_detector --image data/raw/test.jpg --output data/processed/

# Classical HOG+SVM detection
python -m src.classical.detector --image data/raw/test.jpg --output data/processed/
```

### 5. Evaluate and Compare

```bash
python -m src.evaluation.compare --output docs/results/
```

### 6. Interactive Demo

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Repository Structure

```
computer_vision_exam/
├── data/
│   ├── raw/              # Input images and videos
│   ├── processed/        # Preprocessed outputs
│   └── annotations/      # Ground truth labels
├── models/               # YOLO weights, configs, trained classical models
├── src/
│   ├── preprocessing/    # Image blob creation, resizing, normalization
│   ├── classical/        # HOG + SVM pipeline
│   ├── inference/        # YOLO OpenCV DNN wrapper
│   ├── postprocessing/   # IoU and NMS implementations
│   ├── utils/            # Configuration, visualization, video I/O
│   └── evaluation/       # mAP, precision, recall, benchmarking
├── tests/                # Unit tests
├── notebooks/            # Jupyter and Colab demos
├── scripts/              # Automated download helpers
├── docs/
│   ├── project.md        # Original blueprint
│   └── technical_analysis.md  # Source for PDF deliverable
├── requirements.txt
├── environment.yml
├── PROJECT_PLAN.md       # Detailed implementation roadmap
└── README.md
```

---

## Results

| Model | mAP@0.5 | FPS (CPU) | Notes |
|-------|---------|-----------|-------|
| HOG+SVM | -- | -- | Baseline, person-only detection |
| YOLOv3 | -- | -- | 80 COCO classes |
| YOLOv4 | -- | -- | Optional upgrade |

*Results to be populated after experiments.*

---

## Technical Analysis

The full methodology, experimental results, failure analysis, and ethical considerations are documented in:

- **Source:** `docs/technical_analysis.md`
- **PDF:** `docs/technical_analysis.pdf` (generate from markdown)

---

## Ethical Considerations

- **Privacy** — Object detectors in surveillance contexts capture identifiable individuals. Consider face and license plate blurring in production.
- **Bias** — The COCO dataset is Western-centric; performance may degrade on diverse geographies, clothing styles, and underrepresented demographics. Validate on local data where possible.
- **Environmental Impact** — This project uses pre-trained weights to avoid energy-intensive training from scratch. CPU inference is the default baseline.
- **Regulatory Compliance** — Deployed monitoring systems should adhere to GDPR, CCPA, and local privacy laws with transparent data retention policies.

---

## License

[Add your license here]

---

## Acknowledgments

- PJ Reddie for YOLO and Darknet
- OpenCV team for the DNN module
- COCO Consortium for the evaluation benchmark
