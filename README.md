# SuthanaLabComputations

Research code for the Suthana Lab (Duke): rhythm analysis, time-series modeling, and clinical ML experiments on the TR-PTSD cohort.

## Repository layout

```
SuthanaLabComputations/
├─ CAPsClassifier(TRPTSD)/
├─ CosinorRegressionModel(TRPTSD)/
├─ LSTM(TRPTSD)/
├─ LogisticRegreesionModel(TRPTSD)/
├─ LICENSE
└─ README.md
```

## Projects

### 1) CosinorRegressionModel(TRPTSD)
Multiple-component cosinor modeling for daily/weekly rhythmicity. The notebooks/scripts implement cosinor fits, heat-map style summaries, and utilities for dataset checks/comparisons. (This folder contains the cosinor notebooks and helper scripts you’ve been using.)

### 2) CAPsClassifier(TRPTSD)
Classification experiments related to the TR-PTSD cohort (CAPS-related labels/features). Use the provided notebooks or scripts in this folder for feature prep and model training/evaluation.

### 3) LSTM(TRPTSD)
Sequence models for longitudinal recordings (e.g., symptom or wearable time series). Includes notebooks/scripts for data windowing, model definition, and training loops.

### 4) LogisticRegreesionModel(TRPTSD)
Baseline logistic-regression pipelines for tabular features. Useful for sanity checks and feature importance baselines.
