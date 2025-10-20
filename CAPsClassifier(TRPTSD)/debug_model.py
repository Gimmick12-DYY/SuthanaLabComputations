#!/usr/bin/env python3
"""
Debug script to investigate why the models are predicting constant values
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def debug_data_and_features():
    """Debug the data loading and feature selection process"""
    
    # Load one patient's data first
    df = pd.read_csv('CAPsClassifier(TRPTSD)/data/Processed_data/processed_RNS_A_B2_Complete.csv')
    print("=== DATA DEBUGGING ===")
    print(f"Dataset shape: {df.shape}")
    print(f"CAPS_score range: {df['CAPS_score'].min()} to {df['CAPS_score'].max()}")
    print(f"CAPS_score mean: {df['CAPS_score'].mean():.2f}")
    print(f"CAPS_score std: {df['CAPS_score'].std():.2f}")
    
    # Check feature columns
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
    
    available_features = []
    for pattern in feature_patterns:
        if pattern in df.columns:
            available_features.append(pattern)
    
    print(f"\nAvailable features: {len(available_features)}")
    print("Features:", available_features)
    
    # Check feature statistics
    print("\n=== FEATURE STATISTICS ===")
    X = df[available_features].copy()
    y = df['CAPS_score'].values
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"Filled {X[col].isna().sum()} missing values in {col} with median: {median_val}")
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Check for constant features
    constant_features = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            constant_features.append(col)
            print(f"WARNING: {col} is constant (all values: {X[col].iloc[0]})")
    
    if constant_features:
        print(f"Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)
    
    # Check feature ranges
    print("\n=== FEATURE RANGES ===")
    for col in X.columns:
        print(f"{col}: min={X[col].min():.4f}, max={X[col].max():.4f}, std={X[col].std():.4f}")
    
    return X, y, available_features

def test_simple_models(X, y):
    """Test simple sklearn models to see if the issue is with the neural network approach"""
    
    print("\n=== TESTING SIMPLE MODELS ===")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test Random Forest
    print("Testing Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"RF MSE: {rf_mse:.4f}")
    print(f"RF R2: {rf_r2:.4f}")
    print(f"RF predictions range: {rf_pred.min():.2f} to {rf_pred.max():.2f}")
    print(f"RF predictions mean: {rf_pred.mean():.2f}")
    
    # Check feature importance
    feature_importance = rf.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, rf_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True CAPS Score')
    plt.ylabel('Predicted CAPS Score')
    plt.title('Random Forest: True vs Predicted')
    plt.show()
    
    return rf_pred, y_test

if __name__ == "__main__":
    X, y, features = debug_data_and_features()
    test_simple_models(X, y)
