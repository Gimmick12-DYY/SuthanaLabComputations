#!/usr/bin/env python3
"""
Patient-specific CAPS classification model to prevent data leakage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob
import os
from tqdm import tqdm
import time

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

def prepare_features(df):
    """Prepare features for classification"""
    # Use the same features as the original model
    safe_features = [
        "cosinor_mean_amplitude", 
        "cosinor_mean_acrophase", 
        "cosinor_mean_mesor",
        "linearAR_Daily_Fit", 
        "linearAR_Weekly_Avg_Daily_Fit", 
        "linearAR_Predicted", 
        "linearAR_Fit_Residual",
        "Sample Entropy", 
        "weekly_sampen"
    ]
    
    available_features = [f for f in safe_features if f in df.columns]
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
    
    return X, y

def train_patient_specific_models(combined_df, patient_info):
    """Train separate models for each patient"""
    print("\nTraining patient-specific models...")
    
    patient_models = {}
    patient_results = {}
    
    for info in tqdm(patient_info, desc="Training patient models"):
        patient = info['patient']
        
        # Get data for this patient only
        patient_df = combined_df[combined_df['patient_id'] == patient].copy()
        
        if len(patient_df) < 50:  # Skip if too few samples
            print(f"  Skipping {patient}: only {len(patient_df)} samples")
            continue
        
        print(f"\n  Training model for {patient}...")
        
        # Prepare features
        X, y = prepare_features(patient_df)
        
        if len(X.columns) == 0:
            print(f"    Skipping {patient}: no valid features after cleaning")
            continue
        
        print(f"    Features: {list(X.columns)}")
        print(f"    Samples: {len(X)}")
        print(f"    CAPS classes: {len(np.unique(y))}")
        
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
        
        # Store results
        patient_models[patient] = {
            'model': model,
            'scaler': scaler,
            'features': X.columns.tolist()
        }
        
        patient_results[patient] = {
            'accuracy': accuracy,
            'n_samples': len(X),
            'n_classes': len(np.unique(y)),
            'classes': np.unique(y),
            'y_test': y_test,
            'y_pred': y_pred,
            'feature_importance': pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        }
        
        print(f"    Accuracy: {accuracy:.4f}")
    
    return patient_models, patient_results

def plot_patient_results_comparison(patient_results):
    """Plot comparison of results across patients"""
    print("\nGenerating patient comparison plots...")
    
    # Extract data for plotting
    patients = list(patient_results.keys())
    accuracies = [patient_results[p]['accuracy'] for p in patients]
    n_samples = [patient_results[p]['n_samples'] for p in patients]
    n_classes = [patient_results[p]['n_classes'] for p in patients]
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    bars1 = ax1.bar(patients, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Classification Accuracy by Patient', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Sample count comparison
    bars2 = ax2.bar(patients, n_samples, color='lightgreen', alpha=0.7)
    ax2.set_title('Number of Samples by Patient', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars2, n_samples):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(n_samples)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Number of classes comparison
    bars3 = ax3.bar(patients, n_classes, color='lightcoral', alpha=0.7)
    ax3.set_title('Number of CAPS Classes by Patient', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Classes')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars3, n_classes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Accuracy vs Sample Count scatter
    ax4.scatter(n_samples, accuracies, s=100, alpha=0.7, c='purple')
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy vs Sample Count', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add patient labels to scatter points
    for i, patient in enumerate(patients):
        ax4.annotate(patient, (n_samples[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(get_outputs_dir(), 'patient_specific_results_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved patient comparison plot: patient_specific_results_comparison.png")

def plot_individual_patient_results(patient_results):
    """Plot individual results for each patient"""
    print("\nGenerating individual patient plots...")
    
    for patient, results in tqdm(patient_results.items(), desc="Creating individual plots"):
        y_test = results['y_test']
        y_pred = results['y_pred']
        accuracy = results['accuracy']
        
        # Create True vs Predicted plot
        plt.figure(figsize=(10, 8))
        
        # Get unique classes for this patient
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        # Plot each class with different color
        for i, cls in enumerate(unique_classes):
            mask_true = y_test == cls
            mask_pred = y_pred == cls
            
            if np.any(mask_true):
                plt.scatter(y_test[mask_true], y_pred[mask_true], 
                           c=[colors[i]], label=f'CAPS {cls}', alpha=0.7, s=30)
        
        # Plot the ideal line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', 
                label='Perfect Prediction', linewidth=2)
        
        plt.xlabel("True CAPS Score", fontsize=12)
        plt.ylabel("Predicted CAPS Score", fontsize=12)
        plt.title(f"True vs Predicted CAPS Scores - {patient}\nAccuracy: {accuracy:.4f}", 
                 fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(get_outputs_dir(), f"{patient}_patient_specific_true_vs_predicted.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot for {patient}")

def plot_feature_importance_comparison(patient_results):
    """Plot feature importance comparison across patients"""
    print("\nGenerating feature importance comparison...")
    
    # Collect all unique features
    all_features = set()
    for results in patient_results.values():
        all_features.update(results['feature_importance']['feature'].tolist())
    
    all_features = sorted(list(all_features))
    
    # Create importance matrix
    importance_matrix = np.zeros((len(patient_results), len(all_features)))
    patients = list(patient_results.keys())
    
    for i, patient in enumerate(patients):
        feature_importance = patient_results[patient]['feature_importance']
        for j, feature in enumerate(all_features):
            if feature in feature_importance['feature'].values:
                importance_matrix[i, j] = feature_importance[
                    feature_importance['feature'] == feature]['importance'].iloc[0]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(importance_matrix, 
                xticklabels=all_features,
                yticklabels=patients,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Feature Importance'})
    
    plt.title('Feature Importance Comparison Across Patients', fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Patients', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(get_outputs_dir(), 'patient_specific_feature_importance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved feature importance comparison: patient_specific_feature_importance_comparison.png")

def main():
    print("Patient-Specific CAPS Classification Model")
    print("=" * 50)
    
    # Load data
    combined_df, patient_info = load_all_patient_data()
    
    # Print patient info
    print("\nPatient Information:")
    for info in patient_info:
        print(f"  {info['patient']}: {info['samples']} samples, "
              f"CAPS {info['caps_min']:.1f}-{info['caps_max']:.1f} "
              f"(mean: {info['caps_mean']:.1f}, unique: {info['caps_unique']})")
    
    # Train patient-specific models
    patient_models, patient_results = train_patient_specific_models(combined_df, patient_info)
    
    if not patient_results:
        print("No valid patient models trained!")
        return
    
    # Print summary results
    print("\n" + "=" * 50)
    print("PATIENT-SPECIFIC MODEL RESULTS:")
    print("=" * 50)
    
    for patient, results in patient_results.items():
        print(f"\n{patient}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Samples: {results['n_samples']}")
        print(f"  Classes: {results['n_classes']}")
        print(f"  Top 3 features:")
        for i, (_, row) in enumerate(results['feature_importance'].head(3).iterrows()):
            print(f"    {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Generate plots
    plot_patient_results_comparison(patient_results)
    plot_individual_patient_results(patient_results)
    plot_feature_importance_comparison(patient_results)
    
    # Calculate overall statistics
    accuracies = [results['accuracy'] for results in patient_results.values()]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"\n" + "=" * 50)
    print("OVERALL STATISTICS:")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Number of Patients: {len(patient_results)}")
    print(f"Accuracy Range: {min(accuracies):.4f} - {max(accuracies):.4f}")
    
    print(f"\nAll plots saved to the 'outputs' directory!")

if __name__ == "__main__":
    main()
