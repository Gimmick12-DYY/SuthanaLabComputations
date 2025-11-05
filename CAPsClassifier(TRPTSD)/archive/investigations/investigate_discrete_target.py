#!/usr/bin/env python3
"""
Investigate if CAPS_score is actually discrete and should be treated as classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def investigate_discrete_target():
    """Check if CAPS_score is actually discrete"""
    
    # Load data
    df = pd.read_csv('data/Processed_data/processed_RNS_A_B2_Complete.csv')
    
    print("=== INVESTIGATING DISCRETE TARGET ===")
    print(f"Dataset shape: {df.shape}")
    
    # Analyze CAPS_score distribution
    caps_values = df['CAPS_score'].values
    unique_values = np.unique(caps_values)
    
    print(f"CAPS_score analysis:")
    print(f"  Total samples: {len(caps_values)}")
    print(f"  Unique values: {len(unique_values)}")
    print(f"  Unique values: {unique_values}")
    print(f"  Value counts:")
    
    for val in unique_values:
        count = np.sum(caps_values == val)
        percentage = count / len(caps_values) * 100
        print(f"    {val}: {count} samples ({percentage:.1f}%)")
    
    # Check if values are evenly distributed
    print(f"\nDistribution analysis:")
    print(f"  Min: {caps_values.min()}")
    print(f"  Max: {caps_values.max()}")
    print(f"  Mean: {caps_values.mean():.2f}")
    print(f"  Std: {caps_values.std():.2f}")
    
    # Check if values are integers or have specific patterns
    print(f"\nValue pattern analysis:")
    print(f"  Are all values integers? {np.all(caps_values == caps_values.astype(int))}")
    print(f"  Are values evenly spaced? {np.allclose(np.diff(np.sort(unique_values)), np.diff(np.sort(unique_values))[0])}")
    
    # Test as classification problem
    print(f"\n=== TESTING AS CLASSIFICATION PROBLEM ===")
    
    # Use safe features
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
        "Hour_polar",
        "Magnet swipes", 
        "Saturations"
    ]
    
    available_features = [f for f in safe_features if f in df.columns]
    X = df[available_features].copy()
    y = caps_values.astype(int)  # Convert to integers for classification
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)
        print(f"Removed constant features: {constant_features}")
    
    print(f"Using {len(X.columns)} features for classification")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = rf_classifier.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nClassification Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Number of classes: {len(np.unique(y))}")
    
    # Check which classes are being predicted
    unique_preds = np.unique(y_pred)
    unique_true = np.unique(y_test)
    
    print(f"  Predicted classes: {unique_preds}")
    print(f"  True classes: {unique_true}")
    print(f"  All true classes predicted? {set(unique_true).issubset(set(unique_preds))}")
    
    # Feature importance
    feature_importance = rf_classifier.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature importance (classification):")
    print(importance_df)
    
    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Classification)')
    plt.colorbar()
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    
    plt.xlabel('Predicted CAPS Score')
    plt.ylabel('True CAPS Score')
    plt.show()
    
    return accuracy, len(unique_values)

def test_with_different_targets():
    """Test with different target variables to see if the issue persists"""
    
    print(f"\n=== TESTING WITH DIFFERENT TARGETS ===")
    
    # Load data
    df = pd.read_csv('data/Processed_data/processed_RNS_A_B2_Complete.csv')
    
    # Test with CAPS_number instead
    if 'CAPS_number' in df.columns:
        print(f"\nTesting with CAPS_number:")
        caps_number = df['CAPS_number'].values
        unique_numbers = np.unique(caps_number)
        print(f"  CAPS_number unique values: {len(unique_numbers)}")
        print(f"  CAPS_number values: {unique_numbers}")
        
        # Check correlation between CAPS_score and CAPS_number
        correlation = np.corrcoef(df['CAPS_score'].values, caps_number)[0, 1]
        print(f"  Correlation between CAPS_score and CAPS_number: {correlation:.4f}")
        
        if correlation > 0.99:
            print(f"  WARNING: CAPS_score and CAPS_number are nearly identical!")
    
    # Test with a synthetic continuous target
    print(f"\nTesting with synthetic continuous target:")
    np.random.seed(42)
    synthetic_target = np.random.normal(20, 10, len(df))
    print(f"  Synthetic target range: {synthetic_target.min():.2f} to {synthetic_target.max():.2f}")
    print(f"  Synthetic target unique values: {len(np.unique(synthetic_target))}")
    
    # Use same features
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
        "Hour_polar",
        "Magnet swipes", 
        "Saturations"
    ]
    
    available_features = [f for f in safe_features if f in df.columns]
    X = df[available_features].copy()
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        X = X.drop(columns=constant_features)
    
    # Test with synthetic target
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, synthetic_target, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    y_pred = rf.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  Synthetic target R2: {r2:.4f}")
    
    if r2 > 0.9:
        print(f"  WARNING: Even with random target, R2 is very high!")
        print(f"  This suggests there's still data leakage or the model is overfitting")
    else:
        print(f"  This is a more realistic R2 score")

if __name__ == "__main__":
    accuracy, n_classes = investigate_discrete_target()
    test_with_different_targets()
    
    print(f"\n=== CONCLUSION ===")
    if n_classes <= 20:
        print("CAPS_score appears to be discrete with few unique values")
        print("This might be better treated as a classification problem")
        print("The high R2 in regression is likely due to the discrete nature of the target")
    else:
        print("CAPS_score appears to be continuous")
        print("The high R2 might be due to other factors")

