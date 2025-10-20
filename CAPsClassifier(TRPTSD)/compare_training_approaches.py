#!/usr/bin/env python3
"""
Compare individual vs combined training approaches for CAPS prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import glob
import os

def load_all_patient_data():
    """Load and combine all patient data"""
    processed_files = glob.glob('CAPsClassifier(TRPTSD)/data/Processed_data/processed_*.csv')
    
    all_dfs = []
    patient_info = []
    
    for file_path in processed_files:
        patient_name = os.path.basename(file_path).replace('processed_', '').replace('_Complete.csv', '')
        df = pd.read_csv(file_path)
        df['patient_id'] = patient_name
        all_dfs.append(df)
        patient_info.append({
            'patient': patient_name,
            'samples': len(df),
            'caps_range': (df['CAPS_score'].min(), df['CAPS_score'].max()),
            'caps_mean': df['CAPS_score'].mean()
        })
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, patient_info

def prepare_features(df):
    """Prepare features for training"""
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
    
    return X, y

def train_combined_model(combined_df):
    """Train a single model on all subjects combined"""
    print("\n=== TRAINING COMBINED MODEL (All Subjects) ===")
    
    X, y = prepare_features(combined_df)
    print(f"Combined dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
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
    
    print(f"Combined Model - MSE: {mse:.4f}, R2: {r2:.4f}")
    print(f"Predictions range: {y_pred.min():.2f} to {y_pred.max():.2f}")
    
    return rf, scaler, y_test, y_pred, mse, r2

def train_individual_models(combined_df, patient_info):
    """Train separate models for each subject"""
    print("\n=== TRAINING INDIVIDUAL MODELS (Per Subject) ===")
    
    individual_results = []
    
    for info in patient_info:
        patient = info['patient']
        print(f"\nTraining model for {patient}...")
        
        # Get data for this patient
        patient_df = combined_df[combined_df['patient_id'] == patient].copy()
        X, y = prepare_features(patient_df)
        
        if len(X) < 100:  # Skip if too few samples
            print(f"  Skipping {patient}: only {len(X)} samples")
            continue
        
        print(f"  {patient}: {len(X)} samples, {X.shape[1]} features")
        print(f"  CAPS range: {y.min():.1f} to {y.max():.1f}, mean: {y.mean():.1f}")
        
        # Split data
        if len(X) < 20:
            # Use all data for training if very small
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
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
        
        print(f"  {patient} - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        individual_results.append({
            'patient': patient,
            'samples': len(X),
            'mse': mse,
            'r2': r2,
            'model': rf,
            'scaler': scaler,
            'y_test': y_test,
            'y_pred': y_pred
        })
    
    return individual_results

def analyze_results(combined_mse, combined_r2, individual_results):
    """Analyze and compare results"""
    print("\n=== RESULTS COMPARISON ===")
    
    print(f"Combined Model:")
    print(f"  MSE: {combined_mse:.4f}")
    print(f"  R2: {combined_r2:.4f}")
    
    print(f"\nIndividual Models:")
    individual_mse = [r['mse'] for r in individual_results]
    individual_r2 = [r['r2'] for r in individual_results]
    
    print(f"  Average MSE: {np.mean(individual_mse):.4f} ± {np.std(individual_mse):.4f}")
    print(f"  Average R2: {np.mean(individual_r2):.4f} ± {np.std(individual_r2):.4f}")
    print(f"  Best R2: {np.max(individual_r2):.4f} ({individual_results[np.argmax(individual_r2)]['patient']})")
    print(f"  Worst R2: {np.min(individual_r2):.4f} ({individual_results[np.argmin(individual_r2)]['patient']})")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Plot 1: R2 comparison
    plt.subplot(1, 3, 1)
    plt.bar(['Combined'] + [r['patient'] for r in individual_results], 
            [combined_r2] + individual_r2)
    plt.title('R2 Score Comparison')
    plt.ylabel('R2 Score')
    plt.xticks(rotation=45)
    
    # Plot 2: MSE comparison
    plt.subplot(1, 3, 2)
    plt.bar(['Combined'] + [r['patient'] for r in individual_results], 
            [combined_mse] + individual_mse)
    plt.title('MSE Comparison')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)
    
    # Plot 3: Sample size vs performance
    plt.subplot(1, 3, 3)
    plt.scatter([r['samples'] for r in individual_results], individual_r2)
    plt.xlabel('Number of Samples')
    plt.ylabel('R2 Score')
    plt.title('Sample Size vs Performance')
    
    plt.tight_layout()
    plt.show()
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    if combined_r2 > np.mean(individual_r2):
        print("COMBINED TRAINING is better:")
        print("- Higher overall R2 score")
        print("- More robust due to larger dataset")
        print("- Better for general population predictions")
    else:
        print("INDIVIDUAL TRAINING is better:")
        print("- Higher average R2 across subjects")
        print("- More personalized predictions")
        print("- Better for patient-specific care")
    
    # Check for subject-specific patterns
    print(f"\nSubject-specific analysis:")
    for r in individual_results:
        if r['r2'] > 0.8:
            print(f"  {r['patient']}: Excellent prediction (R2={r['r2']:.3f})")
        elif r['r2'] > 0.5:
            print(f"  {r['patient']}: Good prediction (R2={r['r2']:.3f})")
        else:
            print(f"  {r['patient']}: Poor prediction (R2={r['r2']:.3f}) - may need more data")

def main():
    print("Comparing Individual vs Combined Training Approaches")
    print("=" * 60)
    
    # Load data
    combined_df, patient_info = load_all_patient_data()
    
    print("Patient Information:")
    for info in patient_info:
        print(f"  {info['patient']}: {info['samples']} samples, CAPS {info['caps_range'][0]:.1f}-{info['caps_range'][1]:.1f} (mean: {info['caps_mean']:.1f})")
    
    # Train combined model
    combined_model, combined_scaler, combined_y_test, combined_y_pred, combined_mse, combined_r2 = train_combined_model(combined_df)
    
    # Train individual models
    individual_results = train_individual_models(combined_df, patient_info)
    
    # Analyze results
    analyze_results(combined_mse, combined_r2, individual_results)

if __name__ == "__main__":
    main()
