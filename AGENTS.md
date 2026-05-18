# AGENTS.md — Agent Context for computer_vision_exam

> This file is intended for AI coding agents. It describes the target state of the codebase, build/test workflows, and conventions you must follow when modifying this project.

---

## Project Overview

This is a **Real-Time Hand Gesture Recognition Framework** implemented in Python. It compares a classical computer vision pipeline (MediaPipe Hands keypoints + geometric feature engineering + SVM) against a deep-learning pipeline (YOLOv8n hand detection + CNN/ResNet18 gesture classifier) for real-time gesture recognition.

**Theme:** Hand Gesture Recognition (6 classes)  
**Classical Pipeline:** MediaPipe Hands → 21 landmarks → normalized geometric features (distances, angles, ratios) → scikit-learn SVM  
**Deep Learning Pipeline:** YOLOv8n (single-class hand detection) → crop extraction → PyTorch CNN/ResNet18 (6-class gesture classification)  
**Real-time mode:** Webcam stream with switchable `CLASSICAL` (CPU-optimized) and `DEEP` (GPU-recommended) inference pipelines.

The project is organized as an Agile sprint plan (see `PROJECT_PLAN.md`) with conventional scoped commits and GitHub Flow branching.

---

## Technology Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.10 / 3.11 |
| Core CV library | OpenCV >= 4.8.0 (`opencv-python`) |
| Hand landmark detection | `mediapipe` |
| Classical ML / metrics | `scikit-learn`, `scikit-image` |
| Deep learning framework | `torch`, `torchvision`, `ultralytics` (YOLOv8) |
| Data processing | `numpy`, `pandas` |
| Visualization | `matplotlib`, `seaborn` |
| Notebook environment | Jupyter |
| Configuration | `pyyaml` |
| Testing | `pytest` |

### Environment setup

```bash
# Option A: Conda
conda env create -f environment.yml
conda activate cv-exam

# Option B: venv + pip
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Download required artifacts

```bash
# 1. Download HaGRID subset (~2-4GB) — NOT committed to git
python scripts/download_hagrid.py --classes like dislike ok palm fist peace --output data/raw/

# 2. Build structured datasets from raw HaGRID
python scripts/build_yolo_dataset.py
python scripts/build_classifier_dataset.py
python scripts/build_keypoint_dataset.py

# 3. Train models (outputs go to models/ — gitignored)
python scripts/train_svm.py
python scripts/train_yolo.py
python scripts/train_classifier.py
```

`.gitignore` excludes `data/raw/`, `data/processed/`, `models/`, `docs/results/`, and `docs/assets/`. Artifacts are never committed.

---

## Build and Test Commands

### Run tests

```bash
pytest
```

There is no pytest configuration file. Tests are discovered from the `tests/` directory using default pytest behavior.

### Run individual modules

```bash
# Classical inference on a single image
python -m src.classical.detector --image data/raw/test.jpg --output data/processed/

# DL inference (YOLO detect + CNN classify) on a single image
python -m src.deep_learning.inference --image data/raw/test.jpg --output data/processed/

# Comparative evaluation + benchmarking
python -m src.evaluation.compare --output docs/results/

# Webcam live demo
python scripts/run_webcam.py --mode classical --source 0
python scripts/run_webcam.py --mode deep --source 0
```

### Notebooks

```bash
jupyter notebook notebooks/
```

---

## Code Organization

```
src/
├── config.py                   # Centralized config: paths, class names, hyperparameters
├── preprocessing/
│   ├── __init__.py
│   ├── image_pipeline.py       # Resize, normalize, augmentation (OpenCV / Albumentations)
│   └── yolo_formatter.py       # Convert HaGRID annotations → YOLO txt labels
├── classical/
│   ├── __init__.py
│   ├── hand_detector.py        # MediaPipe Hands wrapper (max 2 hands, confidence filter)
│   ├── feature_extractor.py    # 21 keypoints → normalized geometric features
│   ├── gesture_classifier.py   # SVM wrapper
│   └── train.py                # SVM + StandardScaler training pipeline
├── deep_learning/
│   ├── __init__.py
│   ├── yolo_detector.py        # Ultralytics YOLOv8n wrapper
│   ├── gesture_net.py          # Custom CNN or ResNet18 classifier (PyTorch)
│   ├── train_yolo.py           # YOLO fine-tuning script
│   └── train_classifier.py     # CNN training loop
├── postprocessing/
│   ├── __init__.py
│   ├── iou.py                  # Custom IoU (xywh / xyxy) + pairwise matrix
│   └── nms.py                  # Custom NMS + OpenCV NMSBoxes comparison
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Accuracy, Precision, Recall, F1, mAP@0.5, IoU, FPS
│   ├── benchmark.py            # Latency & throughput measurement
│   └── compare.py              # Side-by-side Classical vs DL report
├── webcam/
│   ├── __init__.py
│   ├── streamer.py             # cv2.VideoCapture wrapper
│   └── pipeline.py             # Live loop: acquire → infer → visualize
└── utils/
    ├── __init__.py
    ├── visualization.py        # Draw keypoints, skeleton, bbox, label, FPS, mode indicator
    └── logger.py               # Centralized logging setup
```

All `__init__.py` files are intentionally **empty** — there are no package-level exports. Imports across the project use absolute paths such as `from src.postprocessing.iou import compute_iou`.

### Bounding box contract

Detections follow the unified format:

```python
[class_id, confidence, x, y, w, h]   # xywh normalized or absolute depending on context
```

`postprocessing/iou.py` supports both `xywh` and `xyxy` via the `format=` parameter.

### Configuration

All paths, class names, and hyperparameters live in `src/config.py`. When adding new parameters, place them in the appropriate section (`PREPROCESSING`, `YOLO`, `CLASSICAL`, `CNN`, `EVALUATION`, `VISUALIZATION`, or `WEBCAM`) rather than hard-coding values in module files.

---

## Code Style Guidelines

### Language
- **English only** for code, comments, docstrings, and documentation.

### Docstrings
- Use **Google / NumPy hybrid** style with `Args:` and `Returns:` sections.
- Every module must have a top-level docstring.
- Every public function/method should have a docstring.
- Algorithmic functions (e.g., `manual_nms`, `extract_geometric_features`) should include step-by-step explanations.

### Naming conventions
- Functions, variables, modules: `snake_case`
- Classes: `PascalCase`
- Constants / config dicts: `UPPER_SNAKE_CASE`

### Type hints
- Type hints are **encouraged**.
- Prefer full annotations when adding new functions (e.g., `-> float`, `-> np.ndarray`).
- The project does **not** currently use `typing` module generics (`List`, `Dict`, `Optional`, etc.).

### General practices
- Keep imports absolute (`from src.module import ...`).
- Do not duplicate constants; always import from `src.config`.
- CLI entry points use `argparse` inside a `main()` function guarded by `if __name__ == "__main__":`.
- Handle missing model files gracefully (pattern: `try/except FileNotFoundError` + `pytest.skip()` in tests).

---

## Testing Instructions

### Test structure

```
tests/
├── __init__.py
├── test_iou.py              # Unit tests for compute_iou, compute_iou_matrix
├── test_nms.py              # Unit tests for manual_nms, opencv_nms
├── test_keypoint_features.py # Unit tests for geometric feature extractor
├── test_classifier.py       # Unit tests for SVM and CNN output shapes
└── test_pipeline.py         # Smoke / integration tests on synthetic frames
```

### Test design patterns
- Use simple top-level functions for unit tests.
- Use class-based grouping for integration tests when it improves readability.
- Use synthetic `np.random.randint` images rather than real data to keep tests hermetic.
- If model weights are missing, skip the relevant tests rather than failing the suite.

### Running tests

```bash
pytest -v                  # verbose
pytest tests/test_iou.py   # single file
```

### Coverage gaps
As the project is being refactored, the test suite will grow to cover:
- `postprocessing/iou.py` and `nms.py`
- `classical/feature_extractor.py`
- `deep_learning/gesture_net.py`
- `evaluation/` (metrics, benchmark, compare)
- `webcam/pipeline.py`

When adding features, add corresponding tests in `tests/`.

---

## Development Conventions

### Git workflow
- **Branching:** GitHub Flow. `main` is stable. Create short-lived feature branches: `feat/<scope>-<description>`, `fix/<scope>-<description>`, `docs/<scope>-<description>`.
- **Commits:** Conventional scoped commits: `<type>(<scope>): <description>`.
  - Scopes: `classical`, `dl`, `eval`, `webcam`, `preproc`, `postproc`, `infra`, `docs`, `tests`.
- **Merge:** Merge to `main` only after `pytest` passes.
- Do not commit large binary files. `data/`, `models/`, `docs/results/`, `docs/assets/` are gitignored.
- Preserve directory structure with `.gitkeep` files where needed.

### Formatting / linting
There are no active configuration files for Black, flake8, Ruff, Pylint, or mypy. `.gitignore` references cache directories for these tools, but none are enforced.

### CI/CD
There is no GitHub Actions, GitLab CI, or Jenkins. All testing and execution are local.

---

## Security and Ethical Considerations

- **Privacy:** The webcam pipeline captures video locally. No frames or landmarks are transmitted to external servers. In production, inform users and obtain consent before enabling camera access.
- **Bias:** The HaGRID dataset is predominantly composed of subjects from specific geographic and demographic groups. Performance may degrade on underrepresented skin tones, hand sizes, or cultural gesture variations. Report this limitation honestly in the Technical Analysis.
- **Legal:** Real-time gesture recognition in public or shared spaces may require consent under GDPR, CCPA, or local privacy laws.
- **Model provenance:** YOLOv8 weights are downloaded from the official Ultralytics source. MediaPipe models are downloaded automatically by the `mediapipe` package from Google's servers.

---

## Common Pitfalls for Agents

1. **Trust `PROJECT_PLAN.md` as the source of truth** for architecture decisions. Do not revert to the old Urban Surveillance codebase patterns.
2. **Do not assume model weights exist** in a fresh clone. Run the appropriate `scripts/train_*.py` or provide download instructions. Handle `FileNotFoundError` gracefully.
3. **Do not add package-level imports** to `__init__.py` files unless explicitly required. They are intentionally empty.
4. **Do not hard-code paths.** Always derive paths from `src.config.PROJECT_ROOT` or existing path constants.
5. **Keep the English language** for all code comments, docstrings, and documentation. Legacy Italian content in `Project_steer.md` or old plans should not be extended.
6. **Do not commit the HaGRID dataset or model weights.** These are always excluded by `.gitignore`.
