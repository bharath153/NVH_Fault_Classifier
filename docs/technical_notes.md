# Technical Notes — NVH Fault Classification

## Purpose

This repository demonstrates vibration-signal feature engineering and multiclass classification for drivetrain fault diagnosis.

## Signal Generation

Synthetic signals represent:
- bearing fault impulses;
- gear-mesh modulation;
- mass imbalance;
- healthy vibration.

Noise is added to each generated segment.

## Feature Extraction

The classifier does not use images or a CNN.

Each time-domain segment is transformed into:
- eight FFT frequency-band magnitude features;
- mean;
- standard deviation;
- RMS;
- peak;
- 95th-percentile absolute amplitude.

## Model

A `RandomForestClassifier` is trained on standardized engineered features. PCA is used only for visualization.

## Validation

The workflow includes a held-out train/test split and 5-fold cross-validation.

## Limitations

The data are deliberately structured and synthetic, so very high accuracy should not be interpreted as field performance. Real NVH diagnostics would require measured signals across speed/load conditions, sensor locations, multiple vehicles/components and domain-shift testing.
