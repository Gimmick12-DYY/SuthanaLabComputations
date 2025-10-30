#!/usr/bin/env python3
"""
CAPS Score Classification Model - Minimal 3-Feature Version with Normalized Features

Uses only 3 independent variables:
1. Sample Entropy
2. linearAR_Fit_Residual (CosinorAR residual)
3. cosinor_multiday_r_squared

All features are normalized to 0-1 scale using MinMaxScaler.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime

def get_outputs_dir():
    """Get the outputs directory path with organized subdirectories"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = os.path.join(script_dir, 'outputs', f'run_{timestamp}')
    
    # Create subdirectories
    os.makedirs(os.path.join(outputs_dir, 'summary'), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, 'patient_specific'), exist_ok=True)
    
    return outputs_dir

def load_all_patient_data():
    """Load and combine all patient data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(script_dir, 'data', 'Processed_data')
    processed_files = glob.glob(os.path.join(processed_data_dir, 'processed_*.csv'))
    
    all_dfs = []
    patient_info = []
    
    print("Loading patient data files...")
    for file_path in processed_files:
        patient_name = os.path.basename(file_path).replace('processed_', '').replace('_Complete.csv', '')
        df = pd.read_csv(file_path)
        df['patient_id'] = patient_name
        all_dfs.append(df)
        patient_info.append({
            'patient': patient_name,
            'samples': len(df),
            'caps_unique': df['CAPS_score'].nunique()
        })
        print(f"  Loaded {patient_name}: {len(df)} samples")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, patient_info

def prepare_features(df):
    """Prepare features for classification - only 3 features"""
    # Use only 3 independent features
    required_features = [
        "Sample Entropy",
        "linearAR_Fit_Residual",
        "cosinor_multiday_r_squared"
    ]
    
    # Check if all features exist
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    X = df[required_features].copy()
    y = df['CAPS_score'].values.astype(int)
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
            print(f"  Filled {X[col].isna().sum()} missing values in {col}")
    
    # Check for constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        print(f"  WARNING: Constant features found: {constant_features}")
        X = X.drop(columns=constant_features)
    
    return X, y

def train_classification_model(X, y):
    """Train a Random Forest classification model with normalized features"""
    print(f"\nTraining Random Forest Classifier...")
    print(f"  Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Normalize features to 0-1 scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Features normalized to [0, 1] range")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=10,  # Limit depth to reduce overfitting
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train_scaled, y_train)
    print(f"  Model training complete")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    
    return model, scaler, X_train, X_test, y_train, y_test, y_pred

def plot_feature_importance(model, feature_names, outputs_dir):
    """Plot and save feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for CAPS Score Classification (3 Features)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'summary', 'feature_importance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFeature Importance:")
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df

def plot_confusion_matrix(y_test, y_pred, outputs_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - CAPS Score Classification')
    plt.xlabel('Predicted CAPS Score')
    plt.ylabel('True CAPS Score')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'summary', 'confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_true_vs_predicted(y_test, y_pred, outputs_dir):
    """Plot true vs predicted with jitter"""
    plt.figure(figsize=(10, 8))
    
    # Add jitter for better visualization
    np.random.seed(42)
    y_test_jitter = y_test + np.random.uniform(-0.2, 0.2, len(y_test))
    y_pred_jitter = y_pred + np.random.uniform(-0.2, 0.2, len(y_pred))
    
    plt.scatter(y_test_jitter, y_pred_jitter, alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.5)
    
    # Plot ideal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    plt.xlabel("True CAPS Score", fontsize=12)
    plt.ylabel("Predicted CAPS Score", fontsize=12)
    plt.title("True vs Predicted CAPS Scores (3-Feature Model)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'summary', 'true_vs_predicted.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_class_performance(y_test, y_pred, outputs_dir):
    """Plot per-class performance metrics"""
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    
    class_metrics = []
    for cls in unique_classes:
        mask = y_test == cls
        if np.sum(mask) > 0:
            accuracy = np.sum(y_pred[mask] == cls) / np.sum(mask)
            count = np.sum(mask)
            mae = np.mean(np.abs(y_test[mask] - y_pred[mask]))
            class_metrics.append({
                'class': cls,
                'accuracy': accuracy,
                'count': count,
                'mae': mae
            })
    
    metrics_df = pd.DataFrame(class_metrics)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Accuracy by class
    axes[0].bar(metrics_df['class'], metrics_df['accuracy'], alpha=0.7, color='lightgreen')
    axes[0].set_xlabel('CAPS Score')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by CAPS Score')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Sample count by class
    axes[1].bar(metrics_df['class'], metrics_df['count'], alpha=0.7, color='skyblue')
    axes[1].set_xlabel('CAPS Score')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Test Sample Count by CAPS Score')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: MAE by class
    axes[2].bar(metrics_df['class'], metrics_df['mae'], alpha=0.7, color='salmon')
    axes[2].set_xlabel('CAPS Score')
    axes[2].set_ylabel('Mean Absolute Error')
    axes[2].set_title('MAE by CAPS Score')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'summary', 'per_class_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_classification_report(y_test, y_pred, outputs_dir):
    """Save classification report to text file"""
    report = classification_report(y_test, y_pred)
    
    report_path = os.path.join(outputs_dir, 'summary', 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("CAPS Score Classification Report (3-Feature Model)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Features used:\n")
        f.write("  1. Sample Entropy\n")
        f.write("  2. linearAR_Fit_Residual\n")
        f.write("  3. cosinor_multiday_r_squared\n\n")
        f.write("Normalization: MinMaxScaler (0-1 range)\n\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"\nClassification Report:")
    print(report)

def plot_patient_specific_results(combined_df, model, scaler, feature_names, patient_info, outputs_dir):
    """Generate plots for individual patients (all patients)"""
    print("\nGenerating patient-specific plots (all patients)...")
    
    # Sort patients by sample count (all patients)
    sorted_patients = sorted(patient_info, key=lambda x: x['samples'], reverse=True)
    
    for info in sorted_patients:
        patient = info['patient']
        patient_df = combined_df[combined_df['patient_id'] == patient].copy()
        
        if len(patient_df) < 10:
            continue
        
        # Prepare features
        X_patient = patient_df[feature_names].copy()
        y_patient = patient_df['CAPS_score'].values.astype(int)
        
        # Handle missing values
        for col in X_patient.columns:
            if X_patient[col].isna().any():
                X_patient[col] = X_patient[col].fillna(X_patient[col].median())
        
        # Scale and predict
        X_patient_scaled = scaler.transform(X_patient)
        y_pred_patient = model.predict(X_patient_scaled)
        
        # Calculate metrics
        accuracy = np.sum(y_patient == y_pred_patient) / len(y_patient)
        mae = np.mean(np.abs(y_patient - y_pred_patient))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Add jitter
        np.random.seed(42)
        y_patient_jitter = y_patient + np.random.uniform(-0.15, 0.15, len(y_patient))
        y_pred_jitter = y_pred_patient + np.random.uniform(-0.15, 0.15, len(y_pred_patient))
        
        plt.scatter(y_patient_jitter, y_pred_jitter, alpha=0.6, s=40, c='steelblue', 
                   edgecolors='black', linewidth=0.5)
        
        # Ideal line
        min_val = min(y_patient.min(), y_pred_patient.min())
        max_val = max(y_patient.max(), y_pred_patient.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.xlabel("True CAPS Score", fontsize=12)
        plt.ylabel("Predicted CAPS Score", fontsize=12)
        plt.title(f"{patient} - Accuracy: {accuracy:.3f}, MAE: {mae:.3f}", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'patient_specific', f'{patient}_predictions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  {patient}: Accuracy={accuracy:.4f}, MAE={mae:.4f}")

def save_summary_statistics(y_test, y_pred, outputs_dir):
    """Save summary statistics to file"""
    stats_path = os.path.join(outputs_dir, 'summary', 'summary_statistics.txt')
    
    accuracy = accuracy_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    errors = y_test - y_pred
    
    with open(stats_path, 'w') as f:
        f.write("CAPS Score Classification - Summary Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n")
        f.write(f"RMSE: {np.sqrt(np.mean(errors**2)):.4f}\n\n")
        f.write(f"Test Set Size: {len(y_test)}\n")
        f.write(f"Number of Classes: {len(np.unique(y_test))}\n")
        f.write(f"Perfect Predictions: {np.sum(errors == 0)} ({np.sum(errors == 0)/len(errors)*100:.1f}%)\n\n")
        f.write("Error Statistics:\n")
        f.write(f"  Min Error: {np.min(errors):.2f}\n")
        f.write(f"  Max Error: {np.max(errors):.2f}\n")
        f.write(f"  Std Error: {np.std(errors):.4f}\n")

def main():
    print("=" * 70)
    print("CAPS Score Classification - Minimal 3-Feature Model")
    print("=" * 70)
    print("\nFeatures:")
    print("  1. Sample Entropy")
    print("  2. linearAR_Fit_Residual (CosinorAR residual)")
    print("  3. cosinor_multiday_r_squared")
    print("\nNormalization: MinMaxScaler (0-1 range)")
    print("=" * 70)
    
    # Get outputs directory
    outputs_dir = get_outputs_dir()
    print(f"\nOutputs will be saved to: {outputs_dir}")
    
    # Load data
    combined_df, patient_info = load_all_patient_data()
    print(f"\nCombined dataset: {combined_df.shape[0]} samples from {len(patient_info)} patients")
    
    # Prepare features
    print("\nPreparing features...")
    X, y = prepare_features(combined_df)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Number of unique CAPS scores: {len(np.unique(y))}")
    
    # Train model
    model, scaler, X_train, X_test, y_train, y_test, y_pred = train_classification_model(X, y)
    
    # Generate plots and reports
    print("\n" + "=" * 70)
    print("Generating visualizations and reports...")
    print("=" * 70)
    
    plot_feature_importance(model, X.columns, outputs_dir)
    plot_confusion_matrix(y_test, y_pred, outputs_dir)
    plot_true_vs_predicted(y_test, y_pred, outputs_dir)
    plot_per_class_performance(y_test, y_pred, outputs_dir)
    save_classification_report(y_test, y_pred, outputs_dir)
    save_summary_statistics(y_test, y_pred, outputs_dir)
    
    # Patient-specific results (top 3 only)
    plot_patient_specific_results(combined_df, model, scaler, X.columns, patient_info, outputs_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    accuracy = accuracy_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"\nAll outputs saved to: {outputs_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

