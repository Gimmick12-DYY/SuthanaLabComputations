#!/usr/bin/env python3
"""
Investigate why entropy features are causing data leakage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import glob
import os

def load_all_patient_data():
    """Load and combine all patient data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(script_dir, 'data', 'Processed_data')
    processed_files = glob.glob(os.path.join(processed_data_dir, 'processed_*.csv'))
    
    all_dfs = []
    for file_path in processed_files:
        df = pd.read_csv(file_path)
        patient = os.path.basename(file_path).replace('processed_RNS_', '').replace('_Complete.csv', '')
        df['patient_id'] = patient
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)

def investigate_entropy_caps_relationship(df):
    """Investigate the relationship between entropy and CAPS scores"""
    print("Investigating entropy-CAPS relationship...")
    
    # Check correlations
    entropy_corr = df['Sample Entropy'].corr(df['CAPS_score'])
    weekly_sampen_corr = df['weekly_sampen'].corr(df['CAPS_score'])
    
    print(f"Sample Entropy correlation with CAPS_score: {entropy_corr:.4f}")
    print(f"weekly_sampen correlation with CAPS_score: {weekly_sampen_corr:.4f}")
    
    # Check unique values
    print(f"\nSample Entropy unique values: {df['Sample Entropy'].nunique()}")
    print(f"weekly_sampen unique values: {df['weekly_sampen'].nunique()}")
    print(f"CAPS_score unique values: {df['CAPS_score'].nunique()}")
    
    # Check if entropy values are discrete
    entropy_unique = sorted(df['Sample Entropy'].dropna().unique())
    weekly_sampen_unique = sorted(df['weekly_sampen'].dropna().unique())
    caps_unique = sorted(df['CAPS_score'].dropna().unique())
    
    print(f"\nSample Entropy range: {df['Sample Entropy'].min():.4f} to {df['Sample Entropy'].max():.4f}")
    print(f"weekly_sampen range: {df['weekly_sampen'].min():.4f} to {df['weekly_sampen'].max():.4f}")
    print(f"CAPS_score range: {df['CAPS_score'].min():.1f} to {df['CAPS_score'].max():.1f}")
    
    # Check for exact matches
    print(f"\nChecking for exact matches...")
    
    # Create a mapping of entropy to CAPS
    entropy_caps_mapping = df.groupby('Sample Entropy')['CAPS_score'].agg(['nunique', 'unique']).reset_index()
    entropy_caps_mapping = entropy_caps_mapping[entropy_caps_mapping['nunique'] == 1]
    
    if len(entropy_caps_mapping) > 0:
        print(f"Found {len(entropy_caps_mapping)} Sample Entropy values that map to single CAPS scores!")
        print("This is the source of data leakage!")
        print(entropy_caps_mapping.head(10))
    
    # Check weekly_sampen mapping
    weekly_caps_mapping = df.groupby('weekly_sampen')['CAPS_score'].agg(['nunique', 'unique']).reset_index()
    weekly_caps_mapping = weekly_caps_mapping[weekly_caps_mapping['nunique'] == 1]
    
    if len(weekly_caps_mapping) > 0:
        print(f"\nFound {len(weekly_caps_mapping)} weekly_sampen values that map to single CAPS scores!")
        print("This is also a source of data leakage!")
        print(weekly_caps_mapping.head(10))
    
    return entropy_caps_mapping, weekly_caps_mapping

def check_patient_specific_patterns(df):
    """Check if the leakage is patient-specific"""
    print("\nChecking patient-specific patterns...")
    
    patients = df['patient_id'].unique()
    
    for patient in patients:
        patient_df = df[df['patient_id'] == patient]
        
        # Check entropy-CAPS mapping for this patient
        entropy_caps_mapping = patient_df.groupby('Sample Entropy')['CAPS_score'].agg(['nunique', 'unique']).reset_index()
        entropy_caps_mapping = entropy_caps_mapping[entropy_caps_mapping['nunique'] == 1]
        
        weekly_caps_mapping = patient_df.groupby('weekly_sampen')['CAPS_score'].agg(['nunique', 'unique']).reset_index()
        weekly_caps_mapping = weekly_caps_mapping[weekly_caps_mapping['nunique'] == 1]
        
        print(f"\n{patient}:")
        print(f"  Sample Entropy -> CAPS mappings: {len(entropy_caps_mapping)}")
        print(f"  weekly_sampen -> CAPS mappings: {len(weekly_caps_mapping)}")
        
        if len(entropy_caps_mapping) > 0:
            print(f"  Sample Entropy unique values: {patient_df['Sample Entropy'].nunique()}")
            print(f"  CAPS unique values: {patient_df['CAPS_score'].nunique()}")
            
            # Check if entropy values are nearly 1:1 with CAPS
            if patient_df['Sample Entropy'].nunique() >= patient_df['CAPS_score'].nunique() * 0.8:
                print(f"  WARNING: Sample Entropy has nearly as many unique values as CAPS scores!")

def test_without_entropy(df):
    """Test model without entropy features"""
    print("\nTesting model without entropy features...")
    
    # Use only cosinor features
    features = ['cosinor_mean_amplitude', 'cosinor_mean_acrophase', 'cosinor_mean_mesor']
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) == 0:
        print("No cosinor features available!")
        return
    
    # Prepare data
    X = df[available_features].copy()
    y = df['CAPS_score'].values.astype(int)
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy without entropy features: {accuracy:.4f}")
    print(f"Features used: {available_features}")
    
    if accuracy < 0.7:
        print("SUCCESS: This is a more realistic accuracy!")
    else:
        print("WARNING: Still high accuracy - investigate further")

def main():
    print("Investigating Entropy Data Leakage")
    print("=" * 40)
    
    # Load data
    df = load_all_patient_data()
    print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")
    
    # Investigate entropy-CAPS relationship
    entropy_mapping, weekly_mapping = investigate_entropy_caps_relationship(df)
    
    # Check patient-specific patterns
    check_patient_specific_patterns(df)
    
    # Test without entropy
    test_without_entropy(df)
    
    print("\n" + "=" * 40)
    print("CONCLUSION:")
    
    if len(entropy_mapping) > 0 or len(weekly_mapping) > 0:
        print("DATA LEAKAGE CONFIRMED!")
        print("The entropy features (Sample Entropy, weekly_sampen) contain")
        print("information that directly maps to CAPS scores.")
        print("This is why we get unrealistic accuracy.")
    else:
        print("No obvious 1:1 mapping found, but entropy features still")
        print("show suspiciously high predictive power.")

if __name__ == "__main__":
    main()


