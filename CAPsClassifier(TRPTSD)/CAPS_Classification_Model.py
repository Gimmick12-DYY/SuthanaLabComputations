#!/usr/bin/env python3
"""
CAPS Score Classification Model - Treating CAPS_score as discrete categories

This addresses the issue where CAPS_score has only 18 unique values and should be
treated as classification rather than regression.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_feature_importance(importance_df, filename="feature_importance.png"):
    """Plot and save feature importance"""
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for CAPS Score Classification')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(get_outputs_dir(), filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot: {filename}")

def plot_true_vs_predicted_combined(y_test, y_pred, filename="combined_true_vs_predicted.png"):
    """Plot True vs Predicted for combined dataset with jitter for discrete classes"""
    plt.figure(figsize=(10, 8))
    
    # Add jitter to discrete values for better visualization
    np.random.seed(42)  # For reproducible jitter
    y_test_jitter = y_test + np.random.uniform(-0.2, 0.2, len(y_test))
    y_pred_jitter = y_pred + np.random.uniform(-0.2, 0.2, len(y_pred))
    
    plt.scatter(y_test_jitter, y_pred_jitter, alpha=0.6, s=20, c='blue')
    
    # Plot the ideal line (true == predicted)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    plt.xlabel("True CAPS Score", fontsize=12)
    plt.ylabel("Predicted CAPS Score", fontsize=12)
    plt.title("True vs Predicted CAPS Scores (Combined Dataset)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(get_outputs_dir(), filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined True vs Predicted plot: {filename}")

def plot_individual_patient_results(combined_df, X, y, model, scaler, patient_info):
    """Plot True vs Predicted for each individual patient"""
    print("\nGenerating individual patient plots...")
    
    for info in tqdm(patient_info, desc="Creating patient plots"):
        patient = info['patient']
        
        # Get data for this patient
        patient_df = combined_df[combined_df['patient_id'] == patient].copy()
        
        if len(patient_df) < 10:  # Skip if too few samples
            print(f"  Skipping {patient}: only {len(patient_df)} samples")
            continue
        
        # Prepare features for this patient (same as training)
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
        
        available_features = [f for f in safe_features if f in patient_df.columns]
        X_patient = patient_df[available_features].copy()
        y_patient = patient_df['CAPS_score'].values.astype(int)
        
        # Handle missing values
        for col in X_patient.columns:
            if X_patient[col].isna().any():
                X_patient[col] = X_patient[col].fillna(X_patient[col].median())
        
        # Remove constant features
        constant_features = [col for col in X_patient.columns if X_patient[col].nunique() <= 1]
        if constant_features:
            X_patient = X_patient.drop(columns=constant_features)
        
        if len(X_patient.columns) == 0:
            print(f"  Skipping {patient}: no valid features after cleaning")
            continue
        
        # Ensure feature columns match the training data
        # Add missing columns with zeros and reorder to match training
        for col in X.columns:
            if col not in X_patient.columns:
                X_patient[col] = 0
        
        # Reorder columns to match training data
        X_patient = X_patient[X.columns]
        
        # Scale features
        X_patient_scaled = scaler.transform(X_patient)
        
        # Make predictions
        y_pred_patient = model.predict(X_patient_scaled)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Get unique classes for this patient
        unique_classes = np.unique(np.concatenate([y_patient, y_pred_patient]))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        # Plot each class with different color
        for i, cls in enumerate(unique_classes):
            mask_true = y_patient == cls
            mask_pred = y_pred_patient == cls
            
            if np.any(mask_true):
                plt.scatter(y_patient[mask_true], y_pred_patient[mask_true], 
                           c=[colors[i]], label=f'CAPS {cls}', alpha=0.7, s=30)
        
        # Plot the ideal line
        min_val = min(y_patient.min(), y_pred_patient.min())
        max_val = max(y_patient.max(), y_pred_patient.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', 
                label='Perfect Prediction', linewidth=2)
        
        plt.xlabel("True CAPS Score", fontsize=12)
        plt.ylabel("Predicted CAPS Score", fontsize=12)
        plt.title(f"True vs Predicted CAPS Scores - {patient}", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(get_outputs_dir(), f"{patient}_true_vs_predicted.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot for {patient}")

def plot_correlation_matrix(X, filename="feature_correlation_matrix.png"):
    """Plot and save correlation matrix of features"""
    plt.figure(figsize=(14, 12))
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    
    plt.title("Feature Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(get_outputs_dir(), filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation matrix: {filename}")

def plot_error_analysis(y_test, y_pred, filename="error_analysis.png"):
    """Plot error analysis including residuals, MAE, and R-squared"""
    # Calculate errors
    errors = y_test - y_pred
    abs_errors = np.abs(errors)
    mae = np.mean(abs_errors)
    
    # Calculate R-squared (treating as regression for comparison)
    r2 = r2_score(y_test, y_pred)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Residuals vs Predicted
    axes[0, 0].scatter(y_pred, errors, alpha=0.6, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted CAPS Score')
    axes[0, 0].set_ylabel('Residuals (True - Predicted)')
    axes[0, 0].set_title(f'Residuals vs Predicted\nMAE: {mae:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Absolute Errors Distribution
    axes[0, 1].hist(abs_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(mae, color='r', linestyle='--', linewidth=2, label=f'MAE: {mae:.3f}')
    axes[0, 1].set_xlabel('Absolute Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Absolute Errors')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error vs True Values
    axes[1, 0].scatter(y_test, abs_errors, alpha=0.6, s=20, color='orange')
    axes[1, 0].set_xlabel('True CAPS Score')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Absolute Error vs True CAPS Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance Metrics Summary
    axes[1, 1].axis('off')
    metrics_text = f"""
    Performance Metrics Summary
    
    Mean Absolute Error (MAE): {mae:.4f}
    R-squared (R²): {r2:.4f}
    Root Mean Square Error: {np.sqrt(np.mean(errors**2)):.4f}
    
    Error Statistics:
    Min Error: {np.min(errors):.2f}
    Max Error: {np.max(errors):.2f}
    Std Error: {np.std(errors):.4f}
    
    Perfect Predictions: {np.sum(errors == 0)} / {len(errors)} ({np.sum(errors == 0)/len(errors)*100:.1f}%)
    """
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Error Analysis for CAPS Score Classification', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(get_outputs_dir(), filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved error analysis: {filename}")

def plot_per_class_metrics(y_test, y_pred, filename="per_class_metrics.png"):
    """Plot per-class MAE and accuracy metrics"""
    # Get unique classes
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    
    # Calculate metrics for each class
    class_mae = []
    class_accuracy = []
    class_counts = []
    
    for cls in unique_classes:
        mask = y_test == cls
        if np.sum(mask) > 0:
            class_errors = np.abs(y_test[mask] - y_pred[mask])
            class_mae.append(np.mean(class_errors))
            class_accuracy.append(np.sum(y_pred[mask] == cls) / np.sum(mask))
            class_counts.append(np.sum(mask))
        else:
            class_mae.append(0)
            class_accuracy.append(0)
            class_counts.append(0)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: MAE by Class
    axes[0, 0].bar(range(len(unique_classes)), class_mae, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('CAPS Score Class')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].set_title('MAE by CAPS Score Class')
    axes[0, 0].set_xticks(range(len(unique_classes)))
    axes[0, 0].set_xticklabels(unique_classes, rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy by Class
    axes[0, 1].bar(range(len(unique_classes)), class_accuracy, alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('CAPS Score Class')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy by CAPS Score Class')
    axes[0, 1].set_xticks(range(len(unique_classes)))
    axes[0, 1].set_xticklabels(unique_classes, rotation=45)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Sample Count by Class
    axes[1, 0].bar(range(len(unique_classes)), class_counts, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('CAPS Score Class')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_title('Sample Count by CAPS Score Class')
    axes[1, 0].set_xticks(range(len(unique_classes)))
    axes[1, 0].set_xticklabels(unique_classes, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: MAE vs Sample Count
    axes[1, 1].scatter(class_counts, class_mae, alpha=0.7, s=60, color='red')
    axes[1, 1].set_xlabel('Number of Samples')
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].set_title('MAE vs Sample Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(class_counts, class_mae)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=axes[1, 1].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Per-Class Performance Metrics', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(get_outputs_dir(), filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class metrics: {filename}")

def plot_individual_patient_errors(combined_df, X, y, model, scaler, patient_info, filename_prefix="patient_errors"):
    """Plot error analysis for each individual patient"""
    print("\nGenerating individual patient error plots...")
    
    for info in tqdm(patient_info, desc="Creating patient error plots"):
        patient = info['patient']
        
        # Get data for this patient
        patient_df = combined_df[combined_df['patient_id'] == patient].copy()
        
        if len(patient_df) < 10:  # Skip if too few samples
            continue
        
        # Prepare features for this patient (same as training)
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
        
        available_features = [f for f in safe_features if f in patient_df.columns]
        X_patient = patient_df[available_features].copy()
        y_patient = patient_df['CAPS_score'].values.astype(int)
        
        # Handle missing values
        for col in X_patient.columns:
            if X_patient[col].isna().any():
                X_patient[col] = X_patient[col].fillna(X_patient[col].median())
        
        # Remove constant features
        constant_features = [col for col in X_patient.columns if X_patient[col].nunique() <= 1]
        if constant_features:
            X_patient = X_patient.drop(columns=constant_features)
        
        if len(X_patient.columns) == 0:
            continue
        
        # Ensure feature columns match the training data
        for col in X.columns:
            if col not in X_patient.columns:
                X_patient[col] = 0
        X_patient = X_patient[X.columns]
        
        # Scale features and make predictions
        X_patient_scaled = scaler.transform(X_patient)
        y_pred_patient = model.predict(X_patient_scaled)
        
        # Calculate metrics
        errors = y_patient - y_pred_patient
        abs_errors = np.abs(errors)
        mae = np.mean(abs_errors)
        r2 = r2_score(y_patient, y_pred_patient)
        accuracy = np.sum(y_patient == y_pred_patient) / len(y_patient)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Residuals
        axes[0, 0].scatter(y_pred_patient, errors, alpha=0.6, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted CAPS Score')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title(f'{patient} - Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Absolute Errors Distribution
        axes[0, 1].hist(abs_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(mae, color='r', linestyle='--', linewidth=2, label=f'MAE: {mae:.3f}')
        axes[0, 1].set_xlabel('Absolute Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'{patient} - Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: True vs Predicted with errors
        scatter = axes[1, 0].scatter(y_patient, y_pred_patient, c=abs_errors, 
                                   cmap='Reds', alpha=0.7, s=30)
        min_val = min(y_patient.min(), y_pred_patient.min())
        max_val = max(y_patient.max(), y_pred_patient.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        axes[1, 0].set_xlabel('True CAPS Score')
        axes[1, 0].set_ylabel('Predicted CAPS Score')
        axes[1, 0].set_title(f'{patient} - True vs Predicted (colored by error)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Absolute Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Metrics summary
        axes[1, 1].axis('off')
        metrics_text = f"""
        {patient} Performance Metrics
        
        Accuracy: {accuracy:.4f}
        MAE: {mae:.4f}
        R²: {r2:.4f}
        RMSE: {np.sqrt(np.mean(errors**2)):.4f}
        
        Sample Count: {len(y_patient)}
        Perfect Predictions: {np.sum(errors == 0)} ({np.sum(errors == 0)/len(errors)*100:.1f}%)
        
        CAPS Range: {y_patient.min()} - {y_patient.max()}
        """
        axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'{patient} - Error Analysis', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(get_outputs_dir(), f"{patient}_{filename_prefix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved error plot for {patient}")

def load_all_patient_data():
    """Load and combine all patient data"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(script_dir, 'data', 'Processed_data')
    processed_files = glob.glob(os.path.join(processed_data_dir, 'processed_*.csv'))
    
    all_dfs = []
    patient_info = []
    
    print("Loading patient data files...")
    for i, file_path in enumerate(processed_files, 1):
        patient_name = os.path.basename(file_path).replace('processed_', '').replace('_Complete.csv', '')
        df = pd.read_csv(file_path)
        df['patient_id'] = patient_name
        all_dfs.append(df)
        patient_info.append({
            'patient': patient_name,
            'samples': len(df),
            'caps_range': (df['CAPS_score'].min(), df['CAPS_score'].max()),
            'caps_mean': df['CAPS_score'].mean(),
            'caps_unique': df['CAPS_score'].nunique()
        })
        print(f"  [{i}/{len(processed_files)}] Loading {patient_name}...")
        print(f"      Loaded {len(df)} rows")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df, patient_info

def prepare_features(df):
    """Prepare features for classification"""
    # Use only truly independent features (no data leakage)
    # Removed low-importance features: Hour_polar, Magnet swipes, Saturations
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
    y = df['CAPS_score'].values.astype(int)  # Convert to integers for classification
    
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

def train_classification_model(X, y, model_name="RandomForest"):
    """Train a classification model"""
    print(f"\nTraining {model_name} Classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Training model...")
    with tqdm(total=1, desc="Training progress") as pbar:
        model.fit(X_train_scaled, y_train)
        pbar.update(1)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{model_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Number of classes: {len(np.unique(y))}")
    print(f"  Classes: {sorted(np.unique(y))}")
    
    return model, scaler, y_test, y_pred, y_pred_proba, accuracy

def analyze_classification_results(y_test, y_pred, y_pred_proba, model, X_columns):
    """Analyze classification results"""
    
    print(f"\n=== CLASSIFICATION ANALYSIS ===")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - CAPS Score Classification')
    plt.colorbar()
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    
    plt.xlabel('Predicted CAPS Score')
    plt.ylabel('True CAPS Score')
    plt.tight_layout()
    plt.savefig(os.path.join(get_outputs_dir(), "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix: confusion_matrix.png")
    
    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(importance_df)
    
    # Plot and save feature importance
    plot_feature_importance(importance_df)
    
    # Class distribution analysis
    unique_classes = np.unique(y_test)
    print(f"\nClass Distribution Analysis:")
    for cls in unique_classes:
        count = np.sum(y_test == cls)
        percentage = count / len(y_test) * 100
        print(f"  CAPS Score {cls}: {count} samples ({percentage:.1f}%)")

def main():
    print("CAPS Score Classification Model")
    print("=" * 50)
    
    # Load data
    combined_df, patient_info = load_all_patient_data()
    
    print(f"\nCombined dataset shape: {combined_df.shape}")
    
    # Patient information
    print("\nPatient Information:")
    for info in patient_info:
        print(f"  {info['patient']}: {info['samples']} samples, CAPS {info['caps_range'][0]:.1f}-{info['caps_range'][1]:.1f} (mean: {info['caps_mean']:.1f}, unique: {info['caps_unique']})")
    
    # Prepare features
    X, y = prepare_features(combined_df)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of CAPS score classes: {len(np.unique(y))}")
    print(f"CAPS score classes: {sorted(np.unique(y))}")
    
    # Check class distribution
    print(f"\nClass distribution in full dataset:")
    unique_classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        percentage = count / len(y) * 100
        print(f"  CAPS Score {cls}: {count} samples ({percentage:.1f}%)")
    
    # Train model
    model, scaler, y_test, y_pred, y_pred_proba, accuracy = train_classification_model(X, y)
    
    # Analyze results
    analyze_classification_results(y_test, y_pred, y_pred_proba, model, X.columns)
    
    # Generate additional plots
    print(f"\n=== GENERATING ADDITIONAL PLOTS ===")
    
    # Combined True vs Predicted plot
    plot_true_vs_predicted_combined(y_test, y_pred)
    
    # Individual patient plots
    plot_individual_patient_results(combined_df, X, y, model, scaler, patient_info)
    
    # Correlation matrix
    plot_correlation_matrix(X)
    
    # Error analysis plots
    print(f"\n=== GENERATING ERROR ANALYSIS PLOTS ===")
    plot_error_analysis(y_test, y_pred)
    plot_per_class_metrics(y_test, y_pred)
    plot_individual_patient_errors(combined_df, X, y, model, scaler, patient_info)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Classification Accuracy: {accuracy:.4f}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Model successfully classifies CAPS scores into {len(np.unique(y))} categories")
    
    if accuracy > 0.95:
        print("Excellent classification performance!")
    elif accuracy > 0.8:
        print("Good classification performance")
    else:
        print("Moderate classification performance - may need feature engineering")
    
    print(f"\nAll plots saved to the 'outputs' directory!")

if __name__ == "__main__":
    main()
