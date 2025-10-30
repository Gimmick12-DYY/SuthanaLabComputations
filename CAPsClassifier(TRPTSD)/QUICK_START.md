# CAPS Classifier - Quick Start Guide

## 🚀 Recommended: Feature Group Analysis

**Want to know which features work best? Start here:**

```bash
python feature_group_classifiers.py
```

This will train 3 classifiers and tell you which feature type is most predictive:
- LinearAR features (temporal patterns)
- Entropy measures (signal complexity)  
- Cosinor metrics (circadian rhythms)

**Output**: Organized results in `outputs/feature_groups_YYYYMMDD_HHMMSS/`

📖 **Detailed guide**: See `FEATURE_GROUP_ANALYSIS_README.md`

---

## 🎯 Alternative: Minimal 3-Feature Model

**Want a simple baseline with 3 hand-picked features?**

```bash
python minimal_3feature_classifier.py
```

Uses:
- Sample Entropy
- linearAR_Fit_Residual
- cosinor_multiday_r_squared

**Output**: Organized results in `outputs/run_YYYYMMDD_HHMMSS/`

---

## 🧹 Clean Up Old Outputs

```bash
python cleanup_old_outputs.py
```

Removes old messy output files from previous runs. Keeps timestamped directories.

---

## 📚 More Information

- **`USAGE_GUIDE.md`** - Complete guide to all models
- **`FEATURE_GROUP_ANALYSIS_README.md`** - Deep dive into feature groups
- **`outputs/README.md`** - Explanation of output files

---

## 🔍 What's Different From Before?

### Old Approach ❌
- 6 features (possibly too many)
- StandardScaler (z-scores)
- Messy flat output directory
- All patient plots (24+ files)
- Potentially overfitting (>95% accuracy)

### New Approach ✅
- Feature group analysis OR minimal 3 features
- **MinMaxScaler (0-1 normalization)**
- Organized timestamped directories
- Essential plots only
- Model regularization to reduce overfitting
- **All patients included in patient-specific plots**

---

## 📊 Understanding Your Results

### Good Signs ✅
- Accuracy 70-85% (realistic range)
- Similar performance across patients
- Errors are small (predicting nearby CAPS scores)
- Clear feature importance ranking

### Warning Signs ⚠️
- Accuracy >95% (possible overfitting)
- One patient performs much worse
- Large prediction errors (MAE >5)
- All features have similar importance

---

## 🤔 Which Model Should I Use?

| Use Case | Recommended Model |
|----------|-------------------|
| **"Which features should I focus on?"** | `feature_group_classifiers.py` |
| **"I want a simple baseline"** | `minimal_3feature_classifier.py` |
| **"I need the original model"** | `CAPS_Classification_Model.py` |
| **"I want ensemble methods"** | `XGBoost_RanForest_Fusion_Classifier.py` |

---

## 💡 Pro Tips

1. **Always check patient-specific plots** - Model should work for all patients, not just overall
2. **Lower accuracy is OK** - 75% with good generalization > 98% with overfitting
3. **Compare feature groups** - Tells you which measurements to prioritize
4. **Check confusion matrix** - Are errors close to diagonal? (predicting 30 vs 32 is OK, 30 vs 60 is bad)
5. **Read the text reports** - Per-class metrics reveal which CAPS scores are hard to predict

---

## 🆘 Troubleshooting

**Error: "Missing required features"**
- Check that data files are in `data/Processed_data/`
- Verify CSV files have required column names

**"RuntimeWarning: invalid value encountered"**
- Some features may have NaN values
- Script automatically fills with median values

**Plots look wrong/unclear**
- Check image files in output directory
- Open with image viewer, not text editor

**Accuracy seems too high**
- This is why we created these new models!
- Lower accuracy with proper normalization is more realistic

---

## 📞 Next Steps

1. Run `feature_group_classifiers.py`
2. Check `comparison/feature_group_comparison.png`
3. Read `comparison/performance_summary.txt`
4. Review best-performing feature group results
5. Examine patient-specific plots for generalization

**Goal**: Identify which measurement approach (LinearAR/Entropy/Cosinor) is most predictive of CAPS scores.

