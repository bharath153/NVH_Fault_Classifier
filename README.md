# NVH Drivetrain Fault Classifier — FFT Features + Random Forest

> Classifying synthetic drivetrain vibration faults from frequency-band and statistical features

> [!NOTE]
> The repository name is retained for continuity, but the current implementation is **not a CNN**. It uses engineered FFT-band/statistical features and a `RandomForestClassifier`.

## Overview

This project demonstrates a compact NVH fault-classification workflow using synthetic vibration signals for four classes:

- bearing fault;
- gear defect;
- imbalance;
- healthy operation.

The pipeline generates time-domain vibration signals, extracts frequency-band and statistical features, standardizes them, and trains a Random Forest classifier.

## Feature Pipeline

Each signal segment is converted into:
- eight frequency-band magnitude features;
- mean;
- standard deviation;
- RMS;
- peak amplitude;
- 95th-percentile absolute amplitude.

PCA is used only for visualization of the feature space.

## Model

```text
RandomForestClassifier
n_estimators = 200
train/test split = 80/20
5-fold cross-validation
```

## Data Scope

The vibration signals are **synthetic, physics-inspired demonstrations**. The project is not a validated production diagnostic model and does not use measured drivetrain test-bench data.

## Outputs

The script exports:
- class spectra;
- confusion matrix;
- PCA feature-space visualization;
- feature-importance chart;
- cross-validation metrics in `metrics.json`.

## Run

```bash
git clone https://github.com/bharath153/NVH_CNN.git
cd NVH_CNN
pip install -r requirements.txt
python src/train_classifier.py
```

## Repository Structure

```text
NVH_CNN/
├── src/
│   └── train_classifier.py
├── plots/
├── docs/
├── metrics.json
├── requirements.txt
└── README.md
```

## Interpretation

Very high classification accuracy in this repository should be interpreted in the context of a deliberately structured synthetic dataset. It demonstrates the signal-processing and classification workflow, not field performance on real NVH measurements.
