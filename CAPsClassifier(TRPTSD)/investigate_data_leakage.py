#!/usr/bin/env python3
"""
Investigate data leakage in CAPS classification model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import glob
import os

def load_all_patient_data():
    """Load and combine all patient data"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(script_dir, 'data', 'Processed_data')
    processed_files = glob.glob(os.path.join(processed_data_dir, 'processed_*.csv'))
    
    all_dfs = []
    patient_info = []
    
    print("Loading patient data files...")
    for i, file_path in enumerate(processed_files, 1):
        filename = os.path.basename(file_path)
        patient = filename.replace('processed_RNS_', '').replace('_Complete.csv', '')
        
        print(f"  [{i}/{len(processed_files)}] Loading {patient}...")
        df = pd.read_csv(file_path)
        df['patient_id'] = patient
        all_dfs.append(df)
        
        # Get patient info
        caps_scores = df['CAPS_score'].dropna()
        patient_info.append({
            'patient': patient,
            'samples': len(df),
            'caps_min': caps_scores.min(),
            'caps_max': caps_scores.max(),
            'caps_mean': caps_scores.mean(),
            'caps_unique': caps_scores.nunique()
        })
        
        print(f"      Loaded {len(df)} rows")
    
    if not all_dfs:
        raise ValueError("No processed data files found!")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    
    return combined_df, patient_info

def investigate_feature_target_correlations(df):
    """Investigate correlations between features and target"""
    print("\n=== INVESTIGATING FEATURE-TARGET CORRELATIONS ===")
    
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target and patient-specific columns
    exclude_cols = ['CAPS_score', 'CAPS_number', 'patient_id']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Found {len(feature_cols)} numeric features to investigate")
    
    # Calculate correlations with target
    correlations = []
    for col in feature_cols:
        if df[col].notna().sum() > 0:  # Skip if all NaN
            corr = df[col].corr(df['CAPS_score'])
            if not np.isnan(corr):
                correlations.append({
                    'feature': col,
                    'correlation': abs(corr),
                    'correlation_raw': corr,
                    'n_unique': df[col].nunique(),
                    'n_samples': df[col].notna().sum()
                })
    
    # Sort by absolute correlation
    correlations_df = pd.DataFrame(correlations)
    correlations_df = correlations_df.sort_values('correlation', ascending=False)
    
    print("\nTop 20 features by absolute correlation with CAPS_score:")
    print(correlations_df.head(20).to_string(index=False))
    
    # Check for perfect correlations
    perfect_corr = correlations_df[correlations_df['correlation'] > 0.99]
    if len(perfect_corr) > 0:
        print(f"\n⚠️  WARNING: Found {len(perfect_corr)} features with near-perfect correlation (>0.99):")
        print(perfect_corr.to_string(index=False))
    
    # Check for high correlations
    high_corr = correlations_df[correlations_df['correlation'] > 0.9]
    if len(high_corr) > 0:
        print(f"\n⚠️  WARNING: Found {len(high_corr)} features with high correlation (>0.9):")
        print(high_corr.to_string(index=False))
    
    return correlations_df

def check_for_constant_features(df):
    """Check for constant or near-constant features"""
    print("\n=== CHECKING FOR CONSTANT FEATURES ===")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['CAPS_score', 'CAPS_number', 'patient_id']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    constant_features = []
    for col in feature_cols:
        if df[col].notna().sum() > 0:
            n_unique = df[col].nunique()
            if n_unique <= 1:
                constant_features.append(col)
                print(f"  Constant feature: {col} (only {n_unique} unique values)")
            elif n_unique <= 5:
                print(f"  Near-constant feature: {col} (only {n_unique} unique values)")
    
    return constant_features

def check_feature_value_ranges(df):
    """Check if any features have the same value range as CAPS_score"""
    print("\n=== CHECKING FEATURE VALUE RANGES ===")
    
    caps_min, caps_max = df['CAPS_score'].min(), df['CAPS_score'].max()
    caps_unique = df['CAPS_score'].nunique()
    
    print(f"CAPS_score range: {caps_min} to {caps_max} ({caps_unique} unique values)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['CAPS_score', 'CAPS_number', 'patient_id']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    suspicious_features = []
    for col in feature_cols:
        if df[col].notna().sum() > 0:
            col_min, col_max = df[col].min(), df[col].max()
            col_unique = df[col].nunique()
            
            # Check if range and unique count are very similar
            if (abs(col_min - caps_min) < 1 and abs(col_max - caps_max) < 1 and 
                abs(col_unique - caps_unique) < 5):
                suspicious_features.append({
                    'feature': col,
                    'min': col_min,
                    'max': col_max,
                    'unique': col_unique
                })
                print(f"  Suspicious feature: {col} (range: {col_min}-{col_max}, unique: {col_unique})")
    
    return suspicious_features

def test_with_safe_features(df):
    """Test model with only truly safe features"""
    print("\n=== TESTING WITH SAFE FEATURES ONLY ===")
    
    # Define truly safe features (no temporal or derived features)
    safe_features = [
        "cosinor_mean_amplitude", 
        "cosinor_mean_acrophase", 
        "cosinor_mean_mesor",
        "Sample Entropy", 
        "weekly_sampen"
    ]
    
    # Check which features are available
    available_features = [f for f in safe_features if f in df.columns]
    print(f"Available safe features: {available_features}")
    
    if len(available_features) < 3:
        print("⚠️  Too few safe features available!")
        return None
    
    # Prepare data
    X = df[available_features].copy()
    y = df['CAPS_score'].values.astype(int)
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)
        print(f"Removed constant features: {constant_features}")
    
    if len(X.columns) == 0:
        print("⚠️  No valid features after cleaning!")
        return None
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults with safe features only:")
    print(f"  Features used: {list(X.columns)}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Number of classes: {len(np.unique(y))}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature importance:")
    print(feature_importance.to_string(index=False))
    
    return accuracy, feature_importance

def main():
    print("Data Leakage Investigation")
    print("=" * 50)
    
    # Load data
    df, patient_info = load_all_patient_data()
    
    # Print patient info
    print("\nPatient Information:")
    for info in patient_info:
        print(f"  {info['patient']}: {info['samples']} samples, "
              f"CAPS {info['caps_min']:.1f}-{info['caps_max']:.1f} "
              f"(mean: {info['caps_mean']:.1f}, unique: {info['caps_unique']})")
    
    # Investigate correlations
    correlations_df = investigate_feature_target_correlations(df)
    
    # Check for constant features
    constant_features = check_for_constant_features(df)
    
    # Check value ranges
    suspicious_features = check_feature_value_ranges(df)
    
    # Test with safe features
    safe_results = test_with_safe_features(df)
    
    print("\n=== SUMMARY ===")
    print(f"Total features investigated: {len(correlations_df)}")
    print(f"Features with high correlation (>0.9): {len(correlations_df[correlations_df['correlation'] > 0.9])}")
    print(f"Features with perfect correlation (>0.99): {len(correlations_df[correlations_df['correlation'] > 0.99])}")
    print(f"Constant features found: {len(constant_features)}")
    print(f"Suspicious features found: {len(suspicious_features)}")
    
    if safe_results:
        safe_accuracy, _ = safe_results
        print(f"Accuracy with safe features only: {safe_accuracy:.4f}")
        
        if safe_accuracy < 0.8:
            print("SUCCESS: This is a more realistic accuracy!")
        else:
            print("WARNING: Even safe features show high accuracy - investigate further")

if __name__ == "__main__":
    main()
