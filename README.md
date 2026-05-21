# Real-Time Hand Gesture Recognition Framework

A modular computer vision system comparing a classical pipeline (MediaPipe keypoints + geometric features + SVM) against a deep-learning pipeline (YOLOv8 hand detection + ResNet18 gesture classification) for real-time hand gesture recognition.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

---

## Overview

This project implements a complete gesture recognition pipeline for 6 hand gestures:

| Stage | Classical (MediaPipe + SVM) | Deep Learning (YOLOv8 + CNN) |
|-------|----------------------------|------------------------------|
| Preprocessing | Resize, RGB conversion | Resize 640×640, normalization |
| Features | 21 MediaPipe landmarks → geometric features (distances, angles, ratios) | Learned CNN features (ResNet18) |
| Detection | MediaPipe Hands (max 2 hands) | YOLOv8n single-class hand detector |
| Classification | SVM (RBF kernel) | ResNet18 fine-tuned classifier |
| Post-processing | Confidence filter | NMS + confidence filter |
| Evaluation | Accuracy, Precision, Recall, F1 | mAP@0.5, Accuracy, FPS |

### Design Principles
- **Modularity** — Swap backbones without touching visualization or metrics logic
- **Explainability** — Custom IoU and NMS implementations with mathematical transparency
- **Real-time** — Dual-mode webcam pipeline with instant switching (`c` / `d` keys)

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

### 2. Download Dataset

```bash
# Download HaGRID subset (~823MB) — requires Kaggle API token
python scripts/download_hagrid.py --output data/raw/
```

### 3. Build Processed Datasets

```bash
# YOLO format for hand detection
python scripts/build_yolo_dataset.py

# Hand crops for CNN classifier
python scripts/build_classifier_dataset.py

# MediaPipe keypoints + geometric features for SVM
python scripts/build_keypoint_dataset.py
```

### 4. Train Models

```bash
# Classical: MediaPipe + SVM
python scripts/train_svm.py

# Deep Learning: YOLOv8n hand detector
python scripts/train_yolo.py --epochs 20

# Deep Learning: ResNet18 gesture classifier
python scripts/train_classifier.py --epochs 20
```

### 5. Run Inference

```bash
# On a single image — Classical mode
python -m src.classical.hand_detector --image data/raw/test.jpg

# On a single image — Deep mode (requires YOLO + CNN weights)
python -m src.deep_learning.yolo_detector --image data/raw/test.jpg
```

### 6. Live Webcam Demo

```bash
# Classical mode (MediaPipe + SVM, CPU-optimized)
python scripts/run_webcam.py --mode classical --source 0

# Deep mode (YOLO + CNN, GPU recommended)
python scripts/run_webcam.py --mode deep --source 0

# Controls:
#   'c' → switch to CLASSICAL mode
#   'd' → switch to DEEP mode
#   's' → screenshot
#   'q' → quit
```

### 7. Evaluate and Compare

```bash
# On classifier test dataset (with ground-truth labels)
python -m src.evaluation.compare --mode dataset --output docs/results/

# On custom images
python -m src.evaluation.compare --mode images --images image1.jpg image2.jpg --output docs/results/
```

---

## Repository Structure

```
computer_vision_exam/
├── data/
│   ├── raw/                   # HaGRID subset (gitignored)
│   ├── processed/
│   │   ├── yolo/              # YOLO format dataset
│   │   ├── classifier/        # 224×224 hand crops
│   │   └── keypoints/         # CSV features for SVM
│   └── annotations/           # JSON splits
├── models/
│   ├── hand_landmarker.task   # MediaPipe model
│   ├── svm_keypoints.pkl      # Trained SVM + scaler
│   ├── gesture_cnn.pt         # ResNet18 weights
│   └── yolov8n_hand.pt        # Fine-tuned YOLO (gitignored)
├── src/
│   ├── config.py              # Centralized configuration
│   ├── preprocessing/         # Image pipelines
│   ├── classical/             # MediaPipe + SVM pipeline
│   ├── deep_learning/         # YOLOv8 + CNN pipeline
│   ├── postprocessing/        # IoU, NMS
│   ├── evaluation/            # Metrics, benchmark, compare
│   ├── webcam/                # Streamer + live pipeline
│   └── utils/                 # Visualization, logging
├── tests/                     # pytest suite
├── scripts/                   # Entrypoints
├── notebooks/                 # Jupyter demos
├── docs/
│   ├── technical_analysis.md  # Source for PDF deliverable
│   └── results/               # Comparison reports (gitignored)
├── requirements.txt
├── environment.yml
├── PROJECT_PLAN.md            # Implementation roadmap
└── README.md
```

---

## Results

### Classical Pipeline (MediaPipe + SVM)

| Metric | Value |
|--------|-------|
| Validation Accuracy | **91.65%** |
| Precision (macro) | 0.93 |
| Recall (macro) | 0.92 |
| F1 (macro) | 0.92 |
| Inference Time | ~5-10 ms/frame (CPU) |

### Deep Learning Pipeline (YOLOv8 + ResNet18)

| Metric | Value |
|--------|-------|
| CNN Validation Accuracy | **99.69%** |
| CNN Training Epochs | 20 (best at epoch 10) |
| YOLO mAP@0.5 | 0.85 (2 epochs, usable) |
| Inference Time | ~15-30 ms/frame (MPS) |

*Results on HaGRID 30k subset (6 classes: like, dislike, ok, palm, fist, peace).*
*Note: YOLO detector was trained for 2 epochs due to time constraints. It is functional but may miss small hands. For production use, train for 10-20 epochs.*

### End-to-End Evaluation

An end-to-end comparative evaluation was run on the classifier test set (1,562 crops). Due to the crop format (224×224), the **detection rate** varies significantly between pipelines:

| Pipeline | Detection Rate | Accuracy (on detected) |
|----------|---------------|------------------------|
| Classical (MediaPipe+SVM) | ~38% | ~21% |
| Deep (YOLO+CNN) | ~2% | ~21% |

The low detection rate is expected: MediaPipe and YOLO are designed for full-frame images, not pre-cropped hand regions. For meaningful end-to-end metrics, evaluate on full-frame test images or use ground-truth crops directly.

**Validation accuracies (model-level) remain the authoritative metrics:** 91.65% (SVM) and 99.69% (CNN).

---

## Technical Analysis

The full methodology, experimental results, failure analysis, and ethical considerations are documented in:

- **Source:** `docs/technical_analysis.md`
- **PDF:** `docs/technical_analysis.pdf` (auto-generated from markdown)

---

## Ethical Considerations

- **Privacy** — The webcam pipeline captures video locally. No frames or landmarks are transmitted to external servers. Obtain user consent before enabling camera access.
- **Bias** — The HaGRID dataset is predominantly composed of subjects from specific geographic and demographic groups. Performance may degrade on underrepresented skin tones, hand sizes, or cultural gesture variations.
- **Environmental Impact** — This project uses pre-trained weights (ResNet18, YOLOv8n) to avoid energy-intensive training from scratch. Fine-tuning is limited to 20 epochs.
- **Regulatory Compliance** — Real-time gesture recognition in public or shared spaces may require consent under GDPR, CCPA, or local privacy laws.

---

## License

[Add your license here]

---

## Acknowledgments

- [HaGRID](https://github.com/hukenovs/hagrid) dataset authors (Kapitanov et al.)
- Google MediaPipe team for the Hand Landmarker solution
- Ultralytics for YOLOv8
- PyTorch and torchvision teams
