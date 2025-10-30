# CAPS Classifier - Usage Guide

## Overview

This repository contains multiple CAPS score classification approaches. The **recommended** approach for a cleaner, less overfit model is the 3-feature minimal classifier.

## Quick Start - Minimal 3-Feature Classifier

### Run the classifier:
```bash
python minimal_3feature_classifier.py
```

### Features Used:
1. **Sample Entropy** - Measures complexity/randomness in neural activity
2. **linearAR_Fit_Residual** - CosinorAR residual values
3. **cosinor_multiday_r_squared** - Goodness of fit for multi-day cosinor models

### Key Improvements:
- ✅ **Only 3 features** - Reduces overfitting risk
- ✅ **MinMaxScaler normalization** - All features scaled to [0, 1] range
- ✅ **Organized outputs** - Timestamped runs with clear subdirectories
- ✅ **Stratified split** - Maintains class distribution in train/test
- ✅ **Limited tree depth** - Reduces model complexity (max_depth=10)

### Output Structure:
```
outputs/
└── run_YYYYMMDD_HHMMSS/
    ├── summary/                      # Main results
    │   ├── feature_importance.png
    │   ├── confusion_matrix.png
    │   ├── true_vs_predicted.png
    │   ├── per_class_performance.png
    │   ├── classification_report.txt
    │   └── summary_statistics.txt
    └── patient_specific/             # Top 3 patients
        └── RNS_*_predictions.png
```

## Cleanup Old Outputs

To remove legacy output files:
```bash
python cleanup_old_outputs.py
```

This will prompt you before deleting old files. Timestamped run directories are preserved.

## Feature Group Analysis

### Compare Different Feature Types
```bash
python feature_group_classifiers.py
```
Trains **3 separate classifiers** to compare feature groups:
- **LinearAR features**: Daily fit, weekly average, predicted, residual
- **Entropy measures**: Sample entropy, weekly sample entropy
- **Cosinor metrics**: All cosinor-related features (amplitude, acrophase, mesor, r-squared, etc.)

This helps identify which type of features is most predictive of CAPS scores!

Output structure:
```
outputs/feature_groups_YYYYMMDD_HHMMSS/
├── LinearAR/
│   ├── summary/         # LinearAR feature results
│   └── patient_specific/
├── Entropy/
│   ├── summary/         # Entropy feature results
│   └── patient_specific/
├── Cosinor/
│   ├── summary/         # Cosinor feature results
│   └── patient_specific/
└── comparison/          # Side-by-side comparison
    ├── feature_group_comparison.png
    └── performance_summary.txt
```

## Other Models

### Full Feature Model (Original)
```bash
python CAPS_Classification_Model.py
```
- Uses 6 features
- May have higher accuracy but risk of overfitting
- Outputs saved directly to `outputs/` (messy)

### XGBoost + Random Forest Fusion
```bash
python XGBoost_RanForest_Fusion_Classifier.py
```
- Ensemble approach
- More complex but potentially better performance

## Model Comparison

| Model | Features | Normalization | Output Structure | Overfitting Risk |
|-------|----------|---------------|------------------|------------------|
| **minimal_3feature_classifier** | 3 | MinMaxScaler (0-1) | Organized (timestamped) | Low ✓ |
| **feature_group_classifiers** | 3 groups (2-9 each) | MinMaxScaler (0-1) | Organized by group | Low ✓ |
| CAPS_Classification_Model | 6 | MinMaxScaler (0-1) | Flat | Medium |
| XGBoost_RanForest_Fusion | 6+ | StandardScaler | Flat | Medium-High |

## Recommendations

1. **Start with `feature_group_classifiers.py`** to identify which feature types work best
2. Use `minimal_3feature_classifier.py` for a focused 3-feature baseline
3. Check feature importance to understand which features contribute most
4. Review per-class performance to identify problematic CAPS scores
5. Examine patient-specific plots to ensure model generalizes across patients
6. Compare feature group results to determine the most predictive feature category
7. If accuracy is too low, consider combining features from different groups cautiously

## Understanding the Results

### Good Model Indicators:
- Balanced performance across CAPS score classes
- Similar accuracy on different patients
- Confusion matrix shows errors are close to diagonal (predicting nearby CAPS scores)

### Overfitting Indicators:
- Very high accuracy (>95%) with few features
- Large discrepancy between train and test accuracy
- Poor generalization to individual patients

## Data Pipeline

```
Raw Data (data/DataComplete_Cosinor_09.26.25/)
    ↓
Processed Data (data/Processed_data/)
    ↓
Feature Selection (3 features)
    ↓
Normalization (MinMaxScaler)
    ↓
Train/Test Split (80/20, stratified)
    ↓
Random Forest Classifier
    ↓
Evaluation & Visualization
```

## Questions?

Check the outputs/README.md for detailed explanations of each plot type.

