# Feature Group Analysis

## Overview

The `feature_group_classifiers.py` script trains **three separate classifiers** to determine which type of features best predicts CAPS scores. This helps answer the question: *"Which measurement approach is most informative?"*

## Feature Groups

### 1. LinearAR Features (4 features)
- `linearAR_Daily_Fit` - Daily linear AR model fit
- `linearAR_Weekly_Avg_Daily_Fit` - Weekly average of daily fits
- `linearAR_Predicted` - Predicted values from AR model
- `linearAR_Fit_Residual` - Residuals from AR model fit

**Purpose**: Captures temporal patterns and autoregressive behavior in neural activity.

### 2. Entropy Measures (2 features)
- `Sample Entropy` - Daily sample entropy measure
- `weekly_sampen` - Weekly aggregated sample entropy

**Purpose**: Quantifies complexity and irregularity in neural signals.

### 3. Cosinor Metrics (9 features)
- `cosinor_mean_amplitude` - Daily cosinor amplitude
- `cosinor_mean_acrophase` - Daily cosinor phase
- `cosinor_mean_mesor` - Daily cosinor mesor (rhythm-adjusted mean)
- `cosinor_multiday_mesor` - Multi-day cosinor mesor
- `cosinor_multiday_amplitude` - Multi-day cosinor amplitude
- `cosinor_multiday_acrophase_hours` - Multi-day cosinor phase (hours)
- `cosinor_multiday_r_squared` - Multi-day cosinor goodness of fit
- `cosinor_multiday_r_squared_pct` - R² as percentage
- `cosinor_multiday_n` - Number of observations in multi-day window

**Purpose**: Captures circadian rhythms and cyclical patterns in neural activity.

## Output Structure

```
outputs/feature_groups_YYYYMMDD_HHMMSS/
├── LinearAR/
│   ├── summary/
│   │   ├── feature_importance.png        # Which LinearAR features matter most
│   │   ├── confusion_matrix.png          # Prediction patterns
│   │   ├── true_vs_predicted.png         # Scatter plot
│   │   └── classification_report.txt     # Detailed metrics
│   └── patient_specific/
│       ├── RNS_A_B2_predictions.png      # Per-patient results
│       └── ...                           # All patients
├── Entropy/
│   ├── summary/                          # Same structure as LinearAR
│   └── patient_specific/
├── Cosinor/
│   ├── summary/                          # Same structure as LinearAR
│   └── patient_specific/
└── comparison/
    ├── feature_group_comparison.png      # Side-by-side bar charts
    └── performance_summary.txt           # Summary table
```

## Interpreting Results

### Comparison Plots (`comparison/feature_group_comparison.png`)

Three side-by-side bar charts:
1. **Accuracy** - Which feature group achieves highest classification accuracy?
2. **MAE** - Which feature group has lowest prediction error?
3. **Feature Count** - How many features in each group?

### Performance Summary (`comparison/performance_summary.txt`)

Text table showing:
- Feature group name
- Number of features used
- Accuracy on test set
- Mean Absolute Error (MAE)
- Which group performs best

### Individual Group Results

Each feature group folder contains:
- **Feature Importance**: Which specific features within the group matter most
- **Confusion Matrix**: Common prediction errors
- **True vs Predicted**: Visual assessment of prediction quality
- **Patient-Specific Plots**: Does the model work equally well for all patients?

## Expected Outcomes

### Scenario 1: Cosinor Features Dominate
- **Interpretation**: Circadian rhythms strongly correlate with CAPS scores
- **Next Step**: Focus on improving cosinor models, explore different periods

### Scenario 2: Entropy Features Dominate
- **Interpretation**: Signal complexity/irregularity is the key predictor
- **Next Step**: Explore additional entropy measures, optimize window sizes

### Scenario 3: LinearAR Features Dominate
- **Interpretation**: Temporal patterns and autoregressive structure predict CAPS
- **Next Step**: Refine AR model parameters, consider longer time windows

### Scenario 4: Similar Performance
- **Interpretation**: All feature types contribute complementary information
- **Next Step**: Consider combining best features from each group

## Usage

```bash
python feature_group_classifiers.py
```

The script will:
1. Load all patient data
2. Train 3 separate Random Forest classifiers (one per feature group)
3. Generate visualizations for each group
4. Create comparison plots
5. Save all results to timestamped output directory

**Runtime**: ~2-5 minutes depending on data size

## Key Advantages

✅ **Interpretability**: Clearly shows which measurement approach works best  
✅ **Feature Selection**: Guides which features to include in future models  
✅ **Validation**: Ensures conclusions aren't driven by one outlier feature  
✅ **Patient-Level**: Shows if different feature types work better for different patients  

## Troubleshooting

### "Missing features" warning
- Some features may not be available in all datasets
- Script will continue with available features
- Check data preprocessing pipeline

### "No valid features remaining"
- All features in a group are constant (no variation)
- Indicates data quality issue for that feature type
- Review data generation process

### Very high accuracy (>95%)
- Possible overfitting despite regularization
- Check for data leakage
- Verify train/test split is proper

### Very low accuracy (<50%)
- Feature group may not be predictive of CAPS scores
- Normal - not all feature types need to be predictive
- Focus on better-performing groups

## Questions This Analysis Answers

1. **Which feature type is most predictive of CAPS scores?**
   → Compare accuracy/MAE across groups

2. **How many features do we really need?**
   → Entropy group has only 2 features - if it performs well, simpler is better

3. **Are circadian patterns important?**
   → Check Cosinor group performance

4. **Is signal complexity the key?**
   → Check Entropy group performance

5. **Do temporal dependencies matter?**
   → Check LinearAR group performance

6. **Do results generalize across patients?**
   → Review patient-specific plots for each group

