#!/usr/bin/env python3
"""
CAPS Score Classification - Feature Group Analysis

Trains separate classifiers for three feature groups:
1. LinearAR features
2. Entropy measures
3. Cosinor metrics

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

# Define feature groups
FEATURE_GROUPS = {
    'LinearAR': [
        'linearAR_Daily_Fit',
        'linearAR_Weekly_Avg_Daily_Fit',
        'linearAR_Predicted',
        'linearAR_Fit_Residual'
    ],
    'Entropy': [
        'Sample Entropy',
        'weekly_sampen'
    ],
    'Cosinor': [
        'cosinor_multiday_mesor',
        'cosinor_multiday_amplitude',
        'cosinor_multiday_acrophase_hours',
        'cosinor_multiday_r_squared',
        'cosinor_multiday_r_squared_pct',
        'cosinor_multiday_n'
    ]
}

def get_outputs_dir():
    """Get the outputs directory path with organized subdirectories"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = os.path.join(script_dir, 'outputs', f'feature_groups_{timestamp}')
    
    # Create subdirectories for each feature group
    for group_name in FEATURE_GROUPS.keys():
        os.makedirs(os.path.join(outputs_dir, group_name, 'summary'), exist_ok=True)
        os.makedirs(os.path.join(outputs_dir, group_name, 'patient_specific'), exist_ok=True)
    
    os.makedirs(os.path.join(outputs_dir, 'comparison'), exist_ok=True)
    
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

def prepare_features(df, feature_list, group_name):
    """Prepare features for classification"""
    # Check which features are available
    available_features = [f for f in feature_list if f in df.columns]
    missing_features = [f for f in feature_list if f not in df.columns]
    
    if missing_features:
        print(f"  Warning: Missing features in {group_name}: {missing_features}")
    
    if len(available_features) == 0:
        raise ValueError(f"No features available for {group_name}")
    
    print(f"  Using {len(available_features)}/{len(feature_list)} features for {group_name}")
    
    X = df[available_features].copy()
    y = df['CAPS_score'].values.astype(int)
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            missing_count = X[col].isna().sum()
            X[col] = X[col].fillna(X[col].median())
            print(f"    Filled {missing_count} missing values in {col}")
    
    # Remove constant features
    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_features:
        print(f"    Removing constant features: {constant_features}")
        X = X.drop(columns=constant_features)
    
    if len(X.columns) == 0:
        raise ValueError(f"No valid features remaining for {group_name}")
    
    return X, y

def train_model(X, y, group_name):
    """Train a Random Forest classifier"""
    print(f"\n  Training {group_name} Classifier...")
    print(f"    Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features to 0-1 scale
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"    Accuracy: {accuracy:.4f}, MAE: {mae:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'mae': mae,
        'feature_names': X.columns
    }

def plot_feature_importance(results, group_name, outputs_dir):
    """Plot feature importance for a feature group"""
    importance_df = pd.DataFrame({
        'feature': results['feature_names'],
        'importance': results['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, max(6, len(importance_df) * 0.4)))
    plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'{group_name} Features - Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, group_name, 'summary', 'feature_importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(results, group_name, outputs_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(results['y_test'], results['y_pred'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title(f'{group_name} - Confusion Matrix')
    plt.xlabel('Predicted CAPS Score')
    plt.ylabel('True CAPS Score')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, group_name, 'summary', 'confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_true_vs_predicted(results, group_name, outputs_dir):
    """Plot true vs predicted"""
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    plt.figure(figsize=(10, 8))
    
    # Add jitter
    np.random.seed(42)
    y_test_jitter = y_test + np.random.uniform(-0.2, 0.2, len(y_test))
    y_pred_jitter = y_pred + np.random.uniform(-0.2, 0.2, len(y_pred))
    
    plt.scatter(y_test_jitter, y_pred_jitter, alpha=0.5, s=30, c='steelblue',
                edgecolors='black', linewidth=0.5)
    
    # Ideal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.xlabel("True CAPS Score", fontsize=12)
    plt.ylabel("Predicted CAPS Score", fontsize=12)
    plt.title(f"{group_name} - True vs Predicted (Acc: {results['accuracy']:.3f}, MAE: {results['mae']:.3f})",
              fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, group_name, 'summary', 'true_vs_predicted.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

def save_classification_report(results, group_name, outputs_dir):
    """Save classification report"""
    report = classification_report(results['y_test'], results['y_pred'])
    
    report_path = os.path.join(outputs_dir, group_name, 'summary', 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"CAPS Score Classification Report - {group_name} Features\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Features used ({len(results['feature_names'])}):\n")
        for i, feat in enumerate(results['feature_names'], 1):
            f.write(f"  {i}. {feat}\n")
        f.write(f"\nNormalization: MinMaxScaler (0-1 range)\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"MAE: {results['mae']:.4f}\n\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

def plot_patient_specific(combined_df, results, group_name, patient_info, outputs_dir):
    """Generate patient-specific plots"""
    print(f"  Generating patient-specific plots for {group_name}...")
    
    model = results['model']
    scaler = results['scaler']
    feature_names = results['feature_names']
    
    for info in patient_info:
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
        plt.title(f"{group_name} - {patient} (Acc: {accuracy:.3f}, MAE: {mae:.3f})", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, group_name, 'patient_specific', f'{patient}_predictions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def plot_comparison(all_results, outputs_dir):
    """Create comparison plots across feature groups"""
    print("\nGenerating comparison plots...")
    
    # Extract metrics
    group_names = []
    accuracies = []
    maes = []
    feature_counts = []
    
    for group_name, results in all_results.items():
        group_names.append(group_name)
        accuracies.append(results['accuracy'])
        maes.append(results['mae'])
        feature_counts.append(len(results['feature_names']))
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Accuracy comparison
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[0].bar(group_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Feature Group')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (name, acc) in enumerate(zip(group_names, accuracies)):
        axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
    
    # Plot 2: MAE comparison
    axes[1].bar(group_names, maes, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('MAE by Feature Group')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (name, mae) in enumerate(zip(group_names, maes)):
        axes[1].text(i, mae + 0.1, f'{mae:.3f}', ha='center', fontweight='bold')
    
    # Plot 3: Number of features
    axes[2].bar(group_names, feature_counts, color=colors, alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Number of Features')
    axes[2].set_title('Feature Count by Group')
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, (name, count) in enumerate(zip(group_names, feature_counts)):
        axes[2].text(i, count + 0.2, f'{count}', ha='center', fontweight='bold')
    
    plt.suptitle('Feature Group Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'comparison', 'feature_group_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison table
    comparison_path = os.path.join(outputs_dir, 'comparison', 'performance_summary.txt')
    with open(comparison_path, 'w') as f:
        f.write("Feature Group Performance Comparison\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Feature Group':<15} {'Features':<10} {'Accuracy':<12} {'MAE':<10}\n")
        f.write("-" * 70 + "\n")
        for group_name, results in all_results.items():
            f.write(f"{group_name:<15} {len(results['feature_names']):<10} "
                   f"{results['accuracy']:<12.4f} {results['mae']:<10.4f}\n")
        f.write("\n")
        
        # Determine best
        best_acc_group = max(all_results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_mae_group = min(all_results.items(), key=lambda x: x[1]['mae'])[0]
        
        f.write(f"Best Accuracy: {best_acc_group}\n")
        f.write(f"Best MAE: {best_mae_group}\n")

def main():
    print("=" * 70)
    print("CAPS Score Classification - Feature Group Analysis")
    print("=" * 70)
    print("\nFeature Groups:")
    for group_name, features in FEATURE_GROUPS.items():
        print(f"  {group_name}: {len(features)} features")
    print("=" * 70)
    
    # Get outputs directory
    outputs_dir = get_outputs_dir()
    print(f"\nOutputs will be saved to: {outputs_dir}")
    
    # Load data
    combined_df, patient_info = load_all_patient_data()
    print(f"\nCombined dataset: {combined_df.shape[0]} samples from {len(patient_info)} patients")
    
    # Train models for each feature group
    all_results = {}
    
    for group_name, feature_list in FEATURE_GROUPS.items():
        print(f"\n{'='*70}")
        print(f"Processing Feature Group: {group_name}")
        print(f"{'='*70}")
        
        try:
            # Prepare features
            X, y = prepare_features(combined_df, feature_list, group_name)
            
            # Train model
            results = train_model(X, y, group_name)
            all_results[group_name] = results
            
            # Generate visualizations
            print(f"  Generating visualizations for {group_name}...")
            plot_feature_importance(results, group_name, outputs_dir)
            plot_confusion_matrix(results, group_name, outputs_dir)
            plot_true_vs_predicted(results, group_name, outputs_dir)
            save_classification_report(results, group_name, outputs_dir)
            plot_patient_specific(combined_df, results, group_name, patient_info, outputs_dir)
            
        except Exception as e:
            print(f"  ERROR processing {group_name}: {str(e)}")
            continue
    
    # Generate comparison plots
    if len(all_results) > 1:
        plot_comparison(all_results, outputs_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for group_name, results in all_results.items():
        print(f"{group_name:>12}: Accuracy={results['accuracy']:.4f}, MAE={results['mae']:.4f}, "
              f"Features={len(results['feature_names'])}")
    print(f"\nAll outputs saved to: {outputs_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

