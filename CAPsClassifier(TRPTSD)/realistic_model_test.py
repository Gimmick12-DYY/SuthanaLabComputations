#!/usr/bin/env python3
"""
Test with realistic feature selection - no data leakage
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def test_realistic_model():
    """Test with only truly independent features"""
    
    # Load data
    df = pd.read_csv('data/Processed_data/processed_RNS_A_B2_Complete.csv')
    
    print("=== REALISTIC MODEL TEST (No Data Leakage) ===")
    print(f"Dataset shape: {df.shape}")
    print(f"CAPS_score unique values: {df['CAPS_score'].nunique()}")
    print(f"CAPS_score range: {df['CAPS_score'].min()} to {df['CAPS_score'].max()}")
    
    # Use ONLY features that are truly independent of CAPS_score
    # Remove any features that could have exact matches or high correlations
    safe_features = [
        "cosinor_mean_amplitude", 
        "cosinor_mean_acrophase", 
        "cosinor_mean_mesor",
        "linearAR_Daily_Fit", 
        "linearAR_Weekly_Avg_Daily_Fit", 
        "linearAR_Predicted", 
        "linearAR_Fit_Residual",
        "Sample Entropy", 
        "weekly_sampen",
        "Hour_polar",  # Use polar coordinates instead of raw hour
        "Magnet swipes", 
        "Saturations"
    ]
    
    # Check which features are available
    available_features = [f for f in safe_features if f in df.columns]
    print(f"Using {len(available_features)} safe features: {available_features}")
    
    X = df[available_features].copy()
    y = df['CAPS_score'].values
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)
        print(f"Removed constant features: {constant_features}")
    
    print(f"Final feature matrix: {X.shape}")
    
    # Check correlations with target
    print(f"\nFeature correlations with CAPS_score:")
    for col in X.columns:
        correlation = np.corrcoef(X[col].values, y)[0, 1]
        print(f"  {col}: {correlation:.4f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test_scaled)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n=== REALISTIC MODEL RESULTS ===")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Predictions range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    print(f"True range: {y_test.min():.2f} to {y_test.max():.2f}")
    
    # Feature importance
    feature_importance = rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature importance:")
    print(importance_df)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True CAPS Score')
    plt.ylabel('Predicted CAPS Score')
    plt.title(f'Realistic Model: True vs Predicted (R2 = {r2:.4f})')
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuals (True - Predicted)')
    plt.ylabel('Count')
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Check if predictions are reasonable
    print(f"\n=== PREDICTION QUALITY ANALYSIS ===")
    print(f"Mean absolute error: {np.mean(np.abs(y_test - y_pred)):.4f}")
    print(f"Std of residuals: {np.std(residuals):.4f}")
    
    # Check if model is just predicting the mean
    mean_prediction = np.full_like(y_pred, y_train.mean())
    mean_mse = mean_squared_error(y_test, mean_prediction)
    print(f"MSE if predicting mean: {mean_mse:.4f}")
    print(f"Improvement over mean: {((mean_mse - mse) / mean_mse * 100):.1f}%")
    
    return r2, mse

if __name__ == "__main__":
    r2, mse = test_realistic_model()
    
    print(f"\n=== CONCLUSION ===")
    if r2 > 0.8:
        print("Model shows good predictive power with realistic features")
    elif r2 > 0.5:
        print("Model shows moderate predictive power")
    else:
        print("Model shows poor predictive power - may need better features")

