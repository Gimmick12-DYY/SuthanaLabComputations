#!/usr/bin/env python3
"""
Test with minimal features to avoid data leakage
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
from tqdm import tqdm

def get_outputs_dir():
    """Get the outputs directory path"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    return outputs_dir

def load_all_patient_data():
    """Load and combine all patient data"""
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

def test_feature_sets(combined_df, patient_info):
    """Test different feature sets to find the most realistic performance"""
    
    feature_sets = {
        'cosinor_only': ['cosinor_mean_amplitude', 'cosinor_mean_acrophase', 'cosinor_mean_mesor'],
        'linearAR_only': ['linearAR_Daily_Fit', 'linearAR_Weekly_Avg_Daily_Fit', 'linearAR_Predicted', 'linearAR_Fit_Residual'],
        'entropy_only': ['Sample Entropy', 'weekly_sampen'],
        'cosinor_linearAR': ['cosinor_mean_amplitude', 'cosinor_mean_acrophase', 'cosinor_mean_mesor',
                            'linearAR_Daily_Fit', 'linearAR_Weekly_Avg_Daily_Fit', 'linearAR_Predicted', 'linearAR_Fit_Residual'],
        'all_features': ['cosinor_mean_amplitude', 'cosinor_mean_acrophase', 'cosinor_mean_mesor',
                        'linearAR_Daily_Fit', 'linearAR_Weekly_Avg_Daily_Fit', 'linearAR_Predicted', 'linearAR_Fit_Residual',
                        'Sample Entropy', 'weekly_sampen']
    }
    
    results = {}
    
    for set_name, features in feature_sets.items():
        print(f"\nTesting feature set: {set_name}")
        print(f"Features: {features}")
        
        patient_accuracies = []
        
        for info in patient_info:
            patient = info['patient']
            
            # Get data for this patient only
            patient_df = combined_df[combined_df['patient_id'] == patient].copy()
            
            if len(patient_df) < 50:
                continue
            
            # Prepare features
            available_features = [f for f in features if f in patient_df.columns]
            if len(available_features) == 0:
                continue
                
            X = patient_df[available_features].copy()
            y = patient_df['CAPS_score'].values.astype(int)
            
            # Handle missing values
            for col in X.columns:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())
            
            # Remove constant features
            constant_features = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_features:
                X = X.drop(columns=constant_features)
            
            if len(X.columns) == 0:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            patient_accuracies.append(accuracy)
            print(f"  {patient}: {accuracy:.4f}")
        
        if patient_accuracies:
            avg_accuracy = np.mean(patient_accuracies)
            std_accuracy = np.std(patient_accuracies)
            results[set_name] = {
                'avg_accuracy': avg_accuracy,
                'std_accuracy': std_accuracy,
                'patient_accuracies': patient_accuracies,
                'features': features
            }
            print(f"  Average: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return results

def plot_feature_set_comparison(results):
    """Plot comparison of different feature sets"""
    print("\nGenerating feature set comparison plot...")
    
    set_names = list(results.keys())
    avg_accuracies = [results[name]['avg_accuracy'] for name in set_names]
    std_accuracies = [results[name]['std_accuracy'] for name in set_names]
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(set_names, avg_accuracies, yerr=std_accuracies, 
                   capsize=5, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    
    plt.title('Classification Accuracy by Feature Set\n(Patient-Specific Models)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Average Accuracy', fontsize=12)
    plt.xlabel('Feature Set', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc, std in zip(bars, avg_accuracies, std_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(get_outputs_dir(), 'feature_set_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved feature set comparison: feature_set_comparison.png")

def main():
    print("Minimal Features CAPS Classification Test")
    print("=" * 50)
    
    # Load data
    combined_df, patient_info = load_all_patient_data()
    
    # Print patient info
    print("\nPatient Information:")
    for info in patient_info:
        print(f"  {info['patient']}: {info['samples']} samples, "
              f"CAPS {info['caps_min']:.1f}-{info['caps_max']:.1f} "
              f"(mean: {info['caps_mean']:.1f}, unique: {info['caps_unique']})")
    
    # Test different feature sets
    results = test_feature_sets(combined_df, patient_info)
    
    # Generate comparison plot
    plot_feature_set_comparison(results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("FEATURE SET COMPARISON SUMMARY:")
    print("=" * 50)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_accuracy'])
    
    for set_name, result in sorted_results:
        print(f"\n{set_name}:")
        print(f"  Average Accuracy: {result['avg_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
        print(f"  Features: {result['features']}")
        
        if result['avg_accuracy'] < 0.8:
            print("  SUCCESS: REALISTIC PERFORMANCE!")
        elif result['avg_accuracy'] < 0.9:
            print("  WARNING: Still high but more reasonable")
        else:
            print("  ERROR: SUSPICIOUSLY HIGH - likely data leakage")
    
    print(f"\nAll plots saved to the 'outputs' directory!")

if __name__ == "__main__":
    main()
