#!/usr/bin/env python3
"""
Find the exact source of data leakage
"""

import pandas as pd
import numpy as np
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

def test_feature_combinations(df):
    """Test different combinations of features to isolate the leak"""
    print("Testing different feature combinations to find data leakage...")
    
    # Define feature groups
    feature_groups = {
        'cosinor_daily': ['cosinor_mean_amplitude', 'cosinor_mean_acrophase', 'cosinor_mean_mesor'],
        'cosinor_multiday': ['cosinor_multiday_mesor', 'cosinor_multiday_amplitude', 'cosinor_multiday_acrophase_hours', 
                            'cosinor_multiday_r_squared', 'cosinor_multiday_r_squared_pct', 'cosinor_multiday_n'],
        'linearAR': ['linearAR_Daily_Fit', 'linearAR_Weekly_Avg_Daily_Fit', 'linearAR_Predicted', 'linearAR_Fit_Residual'],
        'entropy': ['Sample Entropy', 'weekly_sampen'],
        'patterns': ['Pattern A Channel 1', 'Pattern A Channel 2', 'Pattern B Channel 1', 'Pattern B Channel 2'],
        'episodes': ['Episode starts', 'Episode starts with RX', 'Long episodes'],
        'temporal': ['Month', 'Hour_polar'],
        'other': ['Saturations', 'Magnet swipes']
    }
    
    results = []
    
    # Test each group individually
    for group_name, features in feature_groups.items():
        available_features = [f for f in features if f in df.columns]
        if len(available_features) == 0:
            continue
            
        accuracy = test_features(df, available_features, f"{group_name} only")
        results.append((group_name, accuracy, available_features))
    
    # Test combinations
    combinations = [
        (['cosinor_daily', 'entropy'], 'cosinor_daily + entropy'),
        (['linearAR', 'entropy'], 'linearAR + entropy'),
        (['cosinor_daily', 'linearAR'], 'cosinor_daily + linearAR'),
        (['entropy'], 'entropy only'),
        (['cosinor_daily'], 'cosinor_daily only'),
        (['linearAR'], 'linearAR only'),
    ]
    
    for group_names, combo_name in combinations:
        features = []
        for group_name in group_names:
            features.extend(feature_groups[group_name])
        
        available_features = [f for f in features if f in df.columns]
        if len(available_features) > 0:
            accuracy = test_features(df, available_features, combo_name)
            results.append((combo_name, accuracy, available_features))
    
    # Sort by accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\nResults sorted by accuracy:")
    for group_name, accuracy, features in results:
        print(f"  {group_name}: {accuracy:.4f} ({len(features)} features)")
        if accuracy > 0.95:
            print(f"    WARNING: Suspiciously high accuracy!")
            print(f"    Features: {features}")
    
    return results

def test_features(df, features, test_name):
    """Test a specific set of features"""
    try:
        # Prepare data
        X = df[features].copy()
        y = df['CAPS_score'].values.astype(int)
        
        # Handle missing values
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Remove constant features
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            X = X.drop(columns=constant_features)
        
        if len(X.columns) == 0:
            return 0.0
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)  # Smaller for speed
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
        
    except Exception as e:
        print(f"Error testing {test_name}: {e}")
        return 0.0

def investigate_patient_specific_leakage(df):
    """Check if the leakage is patient-specific"""
    print("\nInvestigating patient-specific leakage...")
    
    patients = df['patient_id'].unique()
    
    for patient in patients:
        patient_df = df[df['patient_id'] == patient]
        
        # Test with only entropy features (should be safe)
        safe_features = ['Sample Entropy', 'weekly_sampen']
        available_features = [f for f in safe_features if f in patient_df.columns]
        
        if len(available_features) > 0:
            accuracy = test_features(patient_df, available_features, f"{patient} entropy only")
            print(f"  {patient}: {accuracy:.4f} (entropy only)")
            
            if accuracy > 0.9:
                print(f"    WARNING: High accuracy for {patient} with entropy only!")

def main():
    print("Finding Data Leakage Source")
    print("=" * 40)
    
    # Load data
    df = load_all_patient_data()
    print(f"Loaded {len(df)} samples from {df['patient_id'].nunique()} patients")
    
    # Test feature combinations
    results = test_feature_combinations(df)
    
    # Investigate patient-specific leakage
    investigate_patient_specific_leakage(df)
    
    print("\n" + "=" * 40)
    print("CONCLUSION:")
    
    # Find the most suspicious result
    max_accuracy = max(results, key=lambda x: x[1])
    if max_accuracy[1] > 0.95:
        print(f"Data leakage found in: {max_accuracy[0]}")
        print(f"Accuracy: {max_accuracy[1]:.4f}")
        print(f"Features: {max_accuracy[2]}")
    else:
        print("No obvious data leakage found in individual feature groups")
        print("The issue might be in the combination of features or patient-specific patterns")

if __name__ == "__main__":
    main()


