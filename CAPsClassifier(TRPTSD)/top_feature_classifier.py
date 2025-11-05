#!/usr/bin/env python3
"""
CAPS Score Classification - Top Feature from Each Group

This classifier:
1. Trains models for each feature group (LinearAR, Entropy, Cosinor)
2. Identifies the most important feature from each group
3. Trains a final classifier using those 3 top features

This provides a focused, interpretable model with features from each measurement approach.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime
from tqdm import tqdm

# Define feature groups (same as feature_group_classifiers.py)
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
    """Get the outputs directory path with timestamp"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = os.path.join(script_dir, 'outputs', f'top_features_{timestamp}')
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

def prepare_features(df, feature_list, group_name):
    """Prepare features for classification"""
    available_features = [f for f in feature_list if f in df.columns]
    missing_features = [f for f in feature_list if f not in df.columns]
    
    if missing_features:
        print(f"  Warning: Missing features in {group_name}: {missing_features}")
    
    if len(available_features) == 0:
        raise ValueError(f"No features available for {group_name}")
    
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
    
    if len(X.columns) == 0:
        raise ValueError(f"No valid features remaining for {group_name}")
    
    return X, y

def get_top_feature_from_group(combined_df, feature_list, group_name):
    """Train a model for a feature group and return the most important feature"""
    print(f"\n  Analyzing {group_name} group...")
    
    # Prepare features
    X, y = prepare_features(combined_df, feature_list, group_name)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
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
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_feature = importance_df.iloc[0]['feature']
    top_importance = importance_df.iloc[0]['importance']
    
    print(f"    Top feature: {top_feature} (importance: {top_importance:.4f})")
    print(f"    All features by importance:")
    for idx, row in importance_df.iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")
    
    return top_feature, importance_df

def train_final_model(combined_df, top_features):
    """Train final classifier using the top features from each group"""
    print(f"\n{'='*70}")
    print("Training Final Classifier with Top Features")
    print(f"{'='*70}")
    print(f"Selected features:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i}. {feat}")
    
    # Prepare features
    X = combined_df[top_features].copy()
    y = combined_df['CAPS_score'].values.astype(int)
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n  Training Random Forest classifier...")
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
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  MAE: {mae:.4f}")
    
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
        'feature_names': top_features
    }

def plot_feature_importance(results, outputs_dir):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'feature': results['feature_names'],
        'importance': results['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top Features - Importance Ranking')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'summary', 'feature_importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved feature importance plot")

def plot_confusion_matrix(results, outputs_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(results['y_test'], results['y_pred'])
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Top Features Classifier')
    plt.xlabel('Predicted CAPS Score')
    plt.ylabel('True CAPS Score')
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'summary', 'confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved confusion matrix")

def plot_true_vs_predicted(results, outputs_dir):
    """Plot true vs predicted"""
    y_test = results['y_test']
    y_pred = results['y_pred']
    
    plt.figure(figsize=(10, 8))
    
    # Add jitter for discrete values
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
    plt.title(f"True vs Predicted (Acc: {results['accuracy']:.3f}, MAE: {results['mae']:.3f})",
              fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'summary', 'true_vs_predicted.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved true vs predicted plot")

def save_classification_report(results, outputs_dir):
    """Save classification report"""
    report = classification_report(results['y_test'], results['y_pred'])
    
    report_path = os.path.join(outputs_dir, 'summary', 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("CAPS Score Classification Report - Top Features from Each Group\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Features used ({len(results['feature_names'])}):\n")
        for i, feat in enumerate(results['feature_names'], 1):
            f.write(f"  {i}. {feat}\n")
        f.write(f"\nNormalization: MinMaxScaler (0-1 range)\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"MAE: {results['mae']:.4f}\n\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    print("  Saved classification report")

def plot_patient_specific(combined_df, results, patient_info, outputs_dir):
    """Generate patient-specific plots"""
    print("\n  Generating patient-specific plots...")
    
    model = results['model']
    scaler = results['scaler']
    feature_names = results['feature_names']
    
    for info in tqdm(patient_info, desc="Patient plots"):
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
        plt.title(f"{patient} (Acc: {accuracy:.3f}, MAE: {mae:.3f})", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'patient_specific', f'{patient}_predictions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("=" * 70)
    print("CAPS Score Classification - Top Feature from Each Group")
    print("=" * 70)
    
    # Get outputs directory
    outputs_dir = get_outputs_dir()
    print(f"\nOutputs will be saved to: {outputs_dir}")
    
    # Load data
    combined_df, patient_info = load_all_patient_data()
    print(f"\nCombined dataset: {combined_df.shape[0]} samples from {len(patient_info)} patients")
    
    # Step 1: Get top feature from each group
    print(f"\n{'='*70}")
    print("Step 1: Identifying Top Features from Each Group")
    print(f"{'='*70}")
    
    top_features = []
    feature_selection_info = {}
    
    for group_name, feature_list in FEATURE_GROUPS.items():
        try:
            top_feature, importance_df = get_top_feature_from_group(combined_df, feature_list, group_name)
            top_features.append(top_feature)
            feature_selection_info[group_name] = {
                'top_feature': top_feature,
                'importance_df': importance_df
            }
        except Exception as e:
            print(f"  ERROR processing {group_name}: {str(e)}")
            continue
    
    if len(top_features) < 3:
        print(f"\nERROR: Could not identify top features from all groups. Only found {len(top_features)} features.")
        return
    
    # Step 2: Train final model with top features
    print(f"\n{'='*70}")
    print("Step 2: Training Final Classifier")
    print(f"{'='*70}")
    
    results = train_final_model(combined_df, top_features)
    
    # Step 3: Generate visualizations
    print(f"\n{'='*70}")
    print("Step 3: Generating Visualizations")
    print(f"{'='*70}")
    
    plot_feature_importance(results, outputs_dir)
    plot_confusion_matrix(results, outputs_dir)
    plot_true_vs_predicted(results, outputs_dir)
    save_classification_report(results, outputs_dir)
    plot_patient_specific(combined_df, results, patient_info, outputs_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Selected Features:")
    for i, (group_name, info) in enumerate(feature_selection_info.items(), 1):
        print(f"  {i}. {group_name}: {info['top_feature']}")
    print(f"\nFinal Model Performance:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"\nAll outputs saved to: {outputs_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

