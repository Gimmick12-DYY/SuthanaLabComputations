#!/usr/bin/env python3
"""
Investigate why we're getting perfect R2 scores - this is suspicious!
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def investigate_data_leakage():
    """Check for data leakage or other issues causing perfect R2"""
    
    # Load one patient's data
    df = pd.read_csv('data/Processed_data/processed_RNS_A_B2_Complete.csv')
    
    print("=== INVESTIGATING PERFECT R2 SCORES ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if CAPS_score is actually the target or if it's being used as a feature
    print(f"\nCAPS_score column info:")
    print(f"  Type: {df['CAPS_score'].dtype}")
    print(f"  Range: {df['CAPS_score'].min()} to {df['CAPS_score'].max()}")
    print(f"  Unique values: {df['CAPS_score'].nunique()}")
    print(f"  Sample values: {df['CAPS_score'].head(10).tolist()}")
    
    # Check for potential data leakage
    print(f"\n=== CHECKING FOR DATA LEAKAGE ===")
    
    # Look for columns that might be identical to CAPS_score
    for col in df.columns:
        if col != 'CAPS_score' and df[col].dtype in ['float64', 'int64']:
            correlation = df['CAPS_score'].corr(df[col])
            if abs(correlation) > 0.99:
                print(f"WARNING: {col} has correlation {correlation:.4f} with CAPS_score!")
                print(f"  CAPS_score sample: {df['CAPS_score'].head(5).tolist()}")
                print(f"  {col} sample: {df[col].head(5).tolist()}")
    
    # Check if CAPS_score is being included as a feature
    feature_patterns = [
        "cosinor_mean_amplitude", "cosinor_mean_acrophase", "cosinor_mean_mesor",
        "cosinor_multiday_mesor", "cosinor_multiday_amplitude", "cosinor_multiday_acrophase_hours",
        "cosinor_multiday_r_squared", "cosinor_multiday_r_squared_pct",
        "linearAR_Daily_Fit", "linearAR_Weekly_Avg_Daily_Fit", "linearAR_Predicted", "linearAR_Fit_Residual",
        "Sample Entropy", "weekly_sampen",
        "Pattern A Channel 1", "Pattern B Channel 1", "Pattern A Channel 2", "Pattern B Channel 2",
        "Episode starts", "Episode starts with RX", "Long episodes",
        "Hour", "Hour_polar", "Hour_degree", "Month",
        "Magnet swipes", "Saturations", "Hist hours", "Mag sat hours"
    ]
    
    available_features = [f for f in feature_patterns if f in df.columns]
    print(f"\nFeatures being used: {available_features}")
    
    # Check if any feature is identical to CAPS_score
    X = df[available_features].copy()
    y = df['CAPS_score'].values
    
    print(f"\n=== CHECKING FEATURE-TARGET RELATIONSHIPS ===")
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            correlation = np.corrcoef(X[col].values, y)[0, 1]
            if abs(correlation) > 0.99:
                print(f"CRITICAL: {col} has correlation {correlation:.6f} with target!")
                print(f"  This means {col} is essentially identical to CAPS_score")
                print(f"  Sample values - {col}: {X[col].head(5).tolist()}")
                print(f"  Sample values - CAPS_score: {y[:5].tolist()}")
    
    # Check for constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        print(f"\nConstant features found: {constant_features}")
        for col in constant_features:
            print(f"  {col}: all values = {X[col].iloc[0]}")
    
    # Check for duplicate rows
    print(f"\n=== CHECKING FOR DUPLICATE DATA ===")
    print(f"Total rows: {len(df)}")
    print(f"Unique rows: {len(df.drop_duplicates())}")
    print(f"Duplicate rows: {len(df) - len(df.drop_duplicates())}")
    
    # Check if CAPS_score has any pattern that might be predictable
    print(f"\n=== CAPS_SCORE PATTERN ANALYSIS ===")
    print(f"CAPS_score value counts (top 10):")
    print(df['CAPS_score'].value_counts().head(10))
    
    # Check if CAPS_score is just a function of other columns
    print(f"\n=== CHECKING FOR DETERMINISTIC RELATIONSHIPS ===")
    
    # Look for exact matches between features and target
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            # Check if any feature values exactly match CAPS_score values
            matches = np.sum(np.abs(X[col].values - y) < 1e-10)
            if matches > 0:
                print(f"WARNING: {col} has {matches} exact matches with CAPS_score")
    
    return df, X, y

def test_without_suspicious_features(df, X, y):
    """Test model performance after removing suspicious features"""
    
    print(f"\n=== TESTING WITH CAREFUL FEATURE SELECTION ===")
    
    # Remove any features that might be causing data leakage
    suspicious_features = []
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            correlation = np.corrcoef(X[col].values, y)[0, 1]
            if abs(correlation) > 0.95:  # Very high correlation
                suspicious_features.append(col)
                print(f"Removing suspicious feature: {col} (correlation: {correlation:.4f})")
    
    # Also remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    suspicious_features.extend(constant_features)
    
    # Remove suspicious features
    clean_features = [col for col in X.columns if col not in suspicious_features]
    X_clean = X[clean_features].copy()
    
    print(f"Original features: {len(X.columns)}")
    print(f"Clean features: {len(X_clean.columns)}")
    print(f"Removed features: {suspicious_features}")
    
    if len(X_clean.columns) == 0:
        print("ERROR: No features left after cleaning!")
        return
    
    # Handle missing values
    for col in X_clean.columns:
        if X_clean[col].isna().any():
            X_clean[col] = X_clean[col].fillna(X_clean[col].median())
    
    # Test with clean features
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    y_pred = rf.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nClean Model Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  Predictions range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    print(f"  True range: {y_test.min():.2f} to {y_test.max():.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True CAPS Score')
    plt.ylabel('Predicted CAPS Score')
    plt.title(f'Clean Model: True vs Predicted (R2 = {r2:.4f})')
    plt.show()
    
    # Check feature importance
    feature_importance = rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_clean.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    print(importance_df.head(10))

if __name__ == "__main__":
    df, X, y = investigate_data_leakage()
    test_without_suspicious_features(df, X, y)
