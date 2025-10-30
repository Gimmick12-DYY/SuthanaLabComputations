# Outputs Directory Structure

## Organization

Each run of the `minimal_3feature_classifier.py` creates a timestamped directory with the following structure:

```
outputs/
├── run_YYYYMMDD_HHMMSS/
│   ├── summary/                      # Main analysis results
│   │   ├── feature_importance.png    # Feature importance bar chart
│   │   ├── confusion_matrix.png      # Confusion matrix heatmap
│   │   ├── true_vs_predicted.png     # Scatter plot of predictions
│   │   ├── per_class_performance.png # Per-class accuracy, MAE, and counts
│   │   ├── classification_report.txt # Detailed classification metrics
│   │   └── summary_statistics.txt    # Overall performance statistics
│   └── patient_specific/             # Individual patient results (top 3)
│       ├── RNS_X_predictions.png     # Per-patient prediction plots
│       └── ...
└── README.md (this file)
```

## Summary Plots

### 1. Feature Importance (`feature_importance.png`)
- Shows the relative importance of the 3 features:
  - Sample Entropy
  - linearAR_Fit_Residual
  - cosinor_multiday_r_squared

### 2. Confusion Matrix (`confusion_matrix.png`)
- Heatmap showing predicted vs true CAPS scores
- Diagonal elements = correct predictions
- Off-diagonal elements = misclassifications

### 3. True vs Predicted (`true_vs_predicted.png`)
- Scatter plot comparing predictions to ground truth
- Red dashed line = perfect prediction
- Points close to line = accurate predictions

### 4. Per-Class Performance (`per_class_performance.png`)
- Three subplots showing:
  - Accuracy by CAPS score class
  - Sample count distribution
  - Mean Absolute Error (MAE) by class

## Text Reports

### Classification Report (`classification_report.txt`)
- Precision, recall, F1-score for each CAPS score
- Macro and weighted averages

### Summary Statistics (`summary_statistics.txt`)
- Overall accuracy
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Error distribution statistics

## Patient-Specific Results

Individual prediction plots for the top 3 patients (by sample count):
- Scatter plot of true vs predicted CAPS scores
- Accuracy and MAE displayed in title
- Helps identify if model performs differently across patients

## Old Outputs

Legacy plots from previous model versions can be found in the root outputs directory.
These can be safely deleted if no longer needed.

