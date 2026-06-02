# Real-Time Hand Gesture Recognition Framework

A modular computer-vision system benchmarking a **classical pipeline** (MediaPipe geometric features + SVM) against a **deep-learning pipeline** (YOLOv8n hand detection + ResNet18 classification) for real-time hand gesture recognition. Built as an assistive-technology prototype to provide a touchless control modality for non-verbal users.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.8+-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

---

## Overview

This project implements a complete, dual-pipeline gesture recognition system for **6 hand gestures**: `like`, `dislike`, `ok`, `palm`, `fist`, `peace`.

| Stage | Classical (MediaPipe + SVM) | Deep Learning (YOLOv8 + CNN) |
|-------|----------------------------|------------------------------|
| Preprocessing | BGR input (MediaPipe handles RGB internally) | YOLO 640×640 (auto-resized by Ultralytics); CNN crops 224×224 + ImageNet normalization |
| Features | 21 MediaPipe landmarks → 54 geometric features (distances, angles, ratios) | Learned CNN features (ResNet18, ImageNet transfer learning) |
| Detection | MediaPipe Hands (max 2 hands) | YOLOv8n single-class hand detector (CSPDarknet + PAN-FPN) |
| Classification | SVM (RBF kernel + StandardScaler) | ResNet18 fine-tuned classifier (512 → 6 classes) |
| Post-processing | MediaPipe detection thresholds | Built-in NMS + confidence filter (custom IoU/NMS tested in `src/postprocessing/`) |
| Evaluation | Accuracy, Precision, Recall, F1 | mAP@0.5, Accuracy, FPS |

### Design Principles
- **Modularity** — Swap backbones without touching visualization or metrics logic.
- **Explainability** — Custom IoU and NMS implementations with mathematical transparency.
- **Real-time** — Dual-mode webcam pipeline with instant switching (`c` / `d` keys).
- **Local-first** — All inference runs on-device; no biometric data leaves the machine.

---

## Problem Statement

Modern Human-Computer Interaction (HCI) is defined by the transition from pixel-level processing to **region-level understanding**. Gesture recognition can be framed as an image-segmentation problem where an image $I$ on domain $\Omega$ is partitioned into regions $\{R_1, R_2, \dots, R_n\}$ such that:

- The union of all regions covers the entire domain: $\bigcup R_i = \Omega$
- The intersection of distinct regions is empty: $R_i \cap R_j = \emptyset$ for $i \neq j$

This formal partitioning allows the system to move beyond raw intensity values to semantic coherence—identifying exactly which pixels belong to the foreground interface (the hand) versus the background environment.

### Technical Challenges
The core difficulty in robust gesture recognition lies in:
- **Intra-class variation** — Significant appearance differences within the same gesture class due to individual hand morphology.
- **Illumination changes** — Shadows and highlights alter pixel values, requiring algorithms to separate reflectance from illumination.
- **Occlusion & ambiguous boundaries** — Self-occlusion of fingers and complex backgrounds make precise edge definition difficult.

This framework addresses these challenges by benchmarking two distinct paradigms: a **Classical Pipeline** based on geometric engineering and a **Deep Learning Pipeline** leveraging neural architectures.

---

## Methodology

### 2.1 Classical Pipeline — Geometric Engineering

The classical approach utilizes the **MediaPipe Hand Landmarker** (Tasks API) for 21-landmark localization in a single pass. These normalized 3D landmarks are transformed into a feature vector comprising **54 elements**:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Wrist distances | 20 | Euclidean distances from the wrist to every other joint |
| Finger segments | 15 | Consecutive joint-to-joint distances across the five fingers |
| PIP angles | 4 | Interior angles at the proximal interphalangeal joints (Cosine Law) |
| Tip-to-tip distances | 10 | Pairwise distances between the five fingertips |
| Finger ratios | 5 | Tip-to-base distance over total bone length (1.0 = straight, ~0.3 = curled) |

All distance features are normalized by palm size (wrist → middle-finger MCP) to ensure **scale invariance**. Classification is performed by a **Support Vector Machine (SVM)** with an **RBF kernel** and **StandardScaler** (zero-center, unit variance).

### 2.2 Deep Learning Pipeline — Neural Architecture

This pipeline follows a two-stage **"detect-then-classify"** paradigm, decoupling localization from semantic interpretation.

- **Detector (YOLOv8n)** — Nano-scale model with a CSPDarknet backbone and PAN-FPN neck for efficient feature aggregation. Employs a decoupled detection head separating classification and regression. Optimized for latency at **8.2 GFLOPs**.
- **Classifier (ResNet18)** — Localization crops (224×224, 10% padding) are passed to a ResNet18 model leveraging **transfer learning** from ImageNet. The final fully-connected layer is replaced to shift output from 512 → 1,000 classes to **512 → 6 gesture classes**, and the full network is fine-tuned. Residual connections mitigate vanishing gradients, allowing the network to learn high-level representations superior to handcrafted features.

> **Spatial Padding:** A 10% padding is applied to all detector crops to prevent finger "amputation" at bounding-box boundaries.

### 2.3 Dataset & Data Integrity

The framework is trained on the [**HaGRID**](https://github.com/hukenovs/hagrid) (HAnd Gesture Recognition Image Dataset) sample 30k source (~10k usable samples after extracting 6 target classes). To prevent data leakage, samples were **stratified by `user_id`** rather than by random image splits, ensuring the model generalizes to *new hands* rather than memorizing specific subjects.

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

### 2. Download Dataset (Optional for inference)

> **Note:** Pre-trained model weights are already included in the repository (`models/`), so you can skip dataset download, dataset building, and training if you only want to run inference or the live demo.

```bash
# Download HaGRID subset (~823 MB) — requires Kaggle API token
python scripts/download_hagrid.py --output data/raw/
```

### 3. Build Processed Datasets (Optional)

```bash
# YOLO format for hand detection
python scripts/build_yolo_dataset.py

# Hand crops for CNN classifier
python scripts/build_classifier_dataset.py

# MediaPipe keypoints + geometric features for SVM
python scripts/build_keypoint_dataset.py
```

### 4. Train Models (Optional)

```bash
# Classical: MediaPipe + SVM
python scripts/train_svm.py

# Deep Learning: YOLOv8n hand detector
python scripts/train_yolo.py --epochs 20

# Deep Learning: ResNet18 gesture classifier
python scripts/train_classifier.py --epochs 50
```

### 5. Live Webcam Demo

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

### 6. Evaluate and Compare

```bash
# On classifier test dataset (with ground-truth labels)
python -m src.evaluation.compare --mode dataset --output docs/results/

# On custom full-frame images
python -m src.evaluation.compare --mode images --images image1.jpg image2.jpg --output docs/results/
```

> **Note:** There is no dedicated single-image inference CLI. Use the webcam demo for live inference or the `compare` module for batch evaluation.

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
│   ├── svm_keypoints.pkl      # Trained SVM + scaler (~1 MB)
│   ├── gesture_cnn.pt         # ResNet18 weights (~45 MB)
│   ├── yolov8n_hand.pt        # Fine-tuned YOLO weights (~23 MB)
│   └── hand_detector/         # YOLO training artifacts
│       └── weights/
│           ├── best.pt
│           └── last.pt
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
├── docs/
│   └── results/               # Comparison reports (gitignored)
├── requirements.txt
├── environment.yml
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
| Inference Time | ~5–10 ms/frame (CPU) |
| Model Size | ~1 MB |

### Deep Learning Pipeline (YOLOv8 + ResNet18)

| Metric | Value |
|--------|-------|
| CNN Validation Accuracy | **99.69%*** |
| CNN Training Epochs | up to 50 (early stopping, patience=5) |
| YOLO mAP@0.5 | **0.995** (20 epochs) |
| YOLO mAP@0.5:0.95 | **0.855** (20 epochs) |
| YOLO Precision | 0.994 |
| YOLO Recall | 0.989 |
| Inference Time | ~15–30 ms/frame (MPS/GPU) |
| Model Size | ~68 MB (23 MB YOLO + 45 MB CNN) |

*Results on HaGRID sample 30k source, ~10k extracted samples (6 classes: like, dislike, ok, palm, fist, peace).*

### Key Findings
- The **deep learning pipeline** delivers a **+7.8% accuracy gain** over the classical approach, making it the preferred choice for high-precision applications.
- The **classical pipeline** remains the optimal choice for edge/mobile deployment where latency and size (~1 MB) are primary constraints.
- End-to-end evaluation on pre-cropped 224×224 images yields artificially low detection rates because both MediaPipe and YOLO are trained on full-frame images. *The 91.65% (SVM) and 99.69% (CNN) accuracies were observed during development on validation sets; they are model-level metrics and should be reproduced on full-frame test images for authoritative benchmarking.*

---

## Failure Analysis

| Failure Mode | Primary Cause | Affected Pipeline |
|--------------|---------------|-------------------|
| "Like" vs. "Palm" confusion | Similar landmark configurations with thumb extended | Classical (MediaPipe) |
| Occlusion errors | Predicted landmarks for hidden joints are often invalid | Classical (MediaPipe) |
| Missed detections | Small hand sizes or sub-optimal training epochs | Deep Learning (YOLO) |
| Crop boundary artifacts | Bounding box cuts off fingers near the edge | Deep Learning (YOLO) |

### Mitigations Implemented
- **Spatial Padding** — 10% padding on all crops prevents finger "amputation."
- **Data Leakage Prevention** — Stratified splits by `user_id` ensure generalization to unseen hands.
- **Geometric Refinement** — Thumb-angle features in the SVM vector help resolve "like" vs. "palm" ambiguities.

---

## Future Directions

1. **Hardware Allocation** — Deploy the classical pipeline for edge/mobile (CPU) and the deep-learning pipeline for high-end GPU environments.
2. **Full Sign-Language Recognition** — Expand beyond the current 6 gestures toward complete ASL/LSF recognition as a comprehensive communication tool for non-verbal users.
3. **Temporal Smoothing** — Implement majority voting over frame sequences to stabilize predictions.
4. **Quantization** — Apply INT8 quantization to the CNN for mobile optimization.
5. **Dataset Diversification** — Audit performance on multi-ethnic datasets to mitigate demographic biases.

---

## Technical Analysis

The full methodology, experimental results, failure analysis, and ethical considerations are documented in:

- **PDF:** [`Technical Analysis_ Real-Time Hand Gesture Recognition Framework.pdf`](./Technical%20Analysis_%20Real-Time%20Hand%20Gesture%20Recognition%20Framework.pdf)

---

## Ethical Considerations

- **Privacy & Local Processing** — The "local-first" architecture ensures that no biometric data or landmarks are transmitted to external servers, aligning with **GDPR** and **CCPA** compliance.
- **Bias & Fairness** — The HaGRID dataset is predominantly Eastern European. Performance may degrade on underrepresented skin tones, age-related morphology (e.g., child-sized hands), or cultural gesture variations.
- **Environmental Impact** — Transfer learning from pre-trained weights (ResNet18, YOLOv8n) avoids energy-intensive training from scratch. Fine-tuning was limited to **20 epochs for YOLO** and **up to 50 epochs for the CNN** (with early stopping, patience=5) to minimize carbon footprint.
- **Regulatory Compliance** — Real-time gesture recognition in public or shared spaces may require explicit consent under GDPR, CCPA, or local privacy laws.

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [HaGRID](https://github.com/hukenovs/hagrid) dataset authors (Kapitanov et al.)
- Google MediaPipe team for the Hand Landmarker solution
- Ultralytics for YOLOv8
- PyTorch and torchvision teams

*Developed by Lucian Claudiu Diaconu — BSc in Computer Engineering & Artificial Intelligence.*
