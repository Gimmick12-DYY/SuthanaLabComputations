# CAPS Classifier - Folder Structure

## 📁 Main Scripts (Production-Ready)

### Primary Classifiers
- **`CAPS_Classification_Model.py`** ⭐ **RECOMMENDED**
  - Automatically selects top feature from each group (LinearAR, Entropy, Cosinor)
  - Trains classifier with 3 most important features
  - Comprehensive analysis and visualizations

- **`top_feature_classifier.py`** ⭐ **NEW**
  - Standalone version that explicitly shows feature selection process
  - Generates detailed logs of which features are selected from each group
  - Same approach as main model but with more detailed output

- **`feature_group_classifiers.py`**
  - Compares performance of different feature groups
  - Trains separate classifiers for each group and compares them
  - Useful for understanding which measurement approach works best

### Alternative Approaches
- **`minimal_3feature_classifier.py`**
  - Uses 3 hand-picked features (Sample Entropy, linearAR_Fit_Residual, cosinor_multiday_r_squared)
  - Simple baseline model

- **`patient_specific_classifier.py`**
  - Trains separate models for each patient
  - Useful for personalized predictions

- **`XGBoost_RanForest_Fusion_Classifier.py`**
  - Advanced ensemble approach using XGBoost and Random Forest

### Utility Scripts
- **`cleanup_old_outputs.py`**
  - Removes old output files while preserving timestamped directories

### Documentation
- **`QUICK_START.md`** - Quick reference guide
- **`USAGE_GUIDE.md`** - Detailed usage instructions
- **`FEATURE_GROUP_ANALYSIS_README.md`** - Feature group analysis guide
- **`FOLDER_STRUCTURE.md`** - This file

---

## 📁 Archive (Investigation Scripts)

Located in `archive/investigations/`:
- `debug_model.py` - Debugging scripts
- `find_data_leakage.py` - Data leakage detection
- `investigate_discrete_target.py` - Discrete target analysis
- `investigate_entropy_leakage.py` - Entropy leakage investigation
- `investigate_perfect_r2.py` - R² investigation
- `realistic_model_test.py` - Model testing
- `compare_training_approaches.py` - Training approach comparison

These scripts were used during development and investigation but are not needed for regular use.

---

## 📁 Outputs

All outputs are saved in the `outputs/` directory with timestamped subdirectories:
- `outputs/top_features_YYYYMMDD_HHMMSS/` - Top feature classifier results
- `outputs/feature_groups_YYYYMMDD_HHMMSS/` - Feature group comparison results
- `outputs/run_YYYYMMDD_HHMMSS/` - Other classifier results

---

## 🚀 Quick Start

**Recommended approach:**
```bash
python CAPS_Classification_Model.py
```

This will:
1. Automatically identify the top feature from each group
2. Train a classifier using those 3 features
3. Generate comprehensive visualizations and reports

**For detailed feature selection logs:**
```bash
python top_feature_classifier.py
```

**To compare feature groups:**
```bash
python feature_group_classifiers.py
```

