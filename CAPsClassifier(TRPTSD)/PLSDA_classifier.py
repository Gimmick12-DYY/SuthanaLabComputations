#!/usr/bin/env python3
"""
CAPS Score Classification - PLS-DA (Partial Least Squares Discriminant Analysis)

This classifier uses PLS-DA to predict CAPS_score using the top feature from each feature group:
- LinearAR features (temporal patterns)
- Entropy measures (signal complexity)
- Cosinor metrics (circadian rhythms)

PLS-DA is particularly useful for:
- Handling multicollinearity
- Dimensionality reduction
- Interpretable latent variables
"""

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime
from tqdm import tqdm

# Define feature groups (same as other classifiers)
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

class PLSDAClassifier:
    """PLS-DA Classifier for multi-class classification"""
    
    def __init__(self, n_components=2, max_iter=500):
        """
        Initialize PLS-DA Classifier
        
        Parameters:
        -----------
        n_components : int, default=2
            Number of PLS components to use
        max_iter : int, default=500
            Maximum number of iterations for PLS
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.pls_models = {}  # One PLS model per class (one-vs-rest)
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Fit PLS-DA model using one-vs-rest approach
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target labels
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train one PLS model per class (one-vs-rest)
        print(f"  Training {len(self.classes_)} PLS models (one-vs-rest)...")
        for i, class_label in enumerate(tqdm(self.classes_, desc="Training PLS models")):
            # Create binary target: 1 for current class, 0 for others
            y_binary = (y_encoded == i).astype(int)
            
            # Train PLS regression
            pls = PLSRegression(n_components=self.n_components, max_iter=self.max_iter)
            pls.fit(X_scaled, y_binary.reshape(-1, 1))
            
            self.pls_models[class_label] = pls
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test features
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each PLS model
        predictions = {}
        for class_label, pls_model in self.pls_models.items():
            y_pred = pls_model.predict(X_scaled)
            predictions[class_label] = y_pred.flatten()
        
        # Convert to DataFrame for easier handling
        pred_df = pd.DataFrame(predictions)
        
        # Predict class with highest score
        y_pred = pred_df.idxmax(axis=1).values
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test features
            
        Returns:
        --------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each PLS model
        predictions = {}
        for class_label, pls_model in self.pls_models.items():
            y_pred = pls_model.predict(X_scaled)
            # Convert to probabilities using sigmoid-like transformation
            # Ensure non-negative and normalize
            y_pred_positive = y_pred.flatten() - y_pred.min() + 1e-10
            predictions[class_label] = y_pred_positive
        
        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Normalize to probabilities
        proba = pred_df.values
        proba = proba / proba.sum(axis=1, keepdims=True)
        
        return proba
    
    def get_latent_variables(self, X):
        """
        Get PLS latent variables (scores) for visualization
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features
            
        Returns:
        --------
        scores : array of shape (n_samples, n_components)
            PLS latent variables
        """
        X_scaled = self.scaler.transform(X)
        # Use first PLS model to get latent variables
        first_model = list(self.pls_models.values())[0]
        scores = first_model.transform(X_scaled)
        return scores

def get_outputs_dir():
    """Get the outputs directory path with timestamp"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs_dir = os.path.join(script_dir, 'outputs', f'plsda_{timestamp}')
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

def get_top_features_from_groups(df):
    """Identify top feature from each feature group based on importance"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    
    top_features = []
    
    for group_name, feature_list in FEATURE_GROUPS.items():
        # Get available features for this group
        available_features = [f for f in feature_list if f in df.columns]
        
        if len(available_features) == 0:
            print(f"  Warning: No features available for {group_name}, skipping...")
            continue
        
        # Prepare features
        X_group = df[available_features].copy()
        y = df['CAPS_score'].values.astype(int)
        
        # Handle missing values
        for col in X_group.columns:
            if X_group[col].isna().any():
                X_group[col] = X_group[col].fillna(X_group[col].median())
        
        # Remove constant features
        constant_features = [col for col in X_group.columns if X_group[col].nunique() <= 1]
        if constant_features:
            X_group = X_group.drop(columns=constant_features)
        
        if len(X_group.columns) == 0:
            continue
        
        # Train a model to get feature importance
        X_train, X_test, y_train, y_test = train_test_split(
            X_group, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X_train_scaled, y_train)
        
        # Get top feature
        importance_df = pd.DataFrame({
            'feature': X_group.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_feature = importance_df.iloc[0]['feature']
        top_importance = importance_df.iloc[0]['importance']
        top_features.append(top_feature)
        
        print(f"  {group_name}: Selected '{top_feature}' (importance: {top_importance:.4f})")
    
    return top_features

def prepare_features(df):
    """Prepare features for classification using top feature from each group"""
    print("\nIdentifying top features from each feature group...")
    top_features = get_top_features_from_groups(df)
    
    if len(top_features) < 3:
        print(f"  Warning: Only found {len(top_features)} top features. Using available features.")
        # Fallback to original features if top feature selection fails
        safe_features = [
            "linearAR_Daily_Fit", 
            "linearAR_Fit_Residual",
            "Sample Entropy",
            "cosinor_multiday_r_squared"
        ]
        available_features = [f for f in safe_features if f in df.columns]
        if len(available_features) == 0:
            raise ValueError("No suitable features found in dataset")
        top_features = available_features[:3]
    
    print(f"\nUsing {len(top_features)} top features:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i}. {feat}")
    
    X = df[top_features].copy()
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
    
    if len(X.columns) == 0:
        raise ValueError("No valid features remaining after cleaning")
    
    return X, y

def train_plsda_model(X, y, n_components=2):
    """Train PLS-DA model"""
    print(f"\nTraining PLS-DA Classifier...")
    print(f"  Number of components: {n_components}")
    print(f"  Number of classes: {len(np.unique(y))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train PLS-DA model
    model = PLSDAClassifier(n_components=n_components, max_iter=500)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'mae': mae,
        'feature_names': X.columns,
        'n_components': n_components
    }

def plot_latent_variables(results, outputs_dir):
    """Plot PLS latent variables (scores)"""
    model = results['model']
    X_test = results['X_test']
    y_test = results['y_test']
    
    # Get latent variables
    scores = model.get_latent_variables(X_test)
    
    # Create plot
    fig, axes = plt.subplots(1, min(3, results['n_components']), figsize=(15, 5))
    if results['n_components'] == 1:
        axes = [axes]
    
    for i in range(min(3, results['n_components'])):
        if i == 0:
            ax = axes[0]
        else:
            ax = axes[i] if len(axes) > i else None
            if ax is None:
                break
        
        # Scatter plot colored by CAPS score
        scatter = ax.scatter(scores[:, i], y_test, c=y_test, cmap='viridis', 
                            alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        ax.set_xlabel(f'PLS Component {i+1}', fontsize=12)
        ax.set_ylabel('CAPS Score', fontsize=12)
        ax.set_title(f'PLS Component {i+1} vs CAPS Score', fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='CAPS Score')
    
    plt.suptitle('PLS Latent Variables', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'summary', 'latent_variables.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved latent variables plot")

def plot_feature_importance(results, outputs_dir):
    """Plot feature importance based on PLS loadings"""
    model = results['model']
    feature_names = results['feature_names']
    
    # Get loadings from first PLS model
    first_model = list(model.pls_models.values())[0]
    loadings = first_model.x_loadings_
    
    # Calculate importance as sum of absolute loadings across components
    importance = np.abs(loadings).sum(axis=1)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance (PLS Loadings)')
    plt.title('Feature Importance - PLS-DA')
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
    plt.title('Confusion Matrix - PLS-DA Classifier')
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
        f.write("CAPS Score Classification Report - PLS-DA\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Features used ({len(results['feature_names'])}):\n")
        for i, feat in enumerate(results['feature_names'], 1):
            f.write(f"  {i}. {feat}\n")
        f.write(f"\nPLS Components: {results['n_components']}\n")
        f.write(f"Normalization: MinMaxScaler (0-1 range)\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"MAE: {results['mae']:.4f}\n\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
    print("  Saved classification report")

def plot_patient_specific(combined_df, results, patient_info, outputs_dir):
    """Generate patient-specific plots"""
    print("\n  Generating patient-specific plots...")
    
    model = results['model']
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
        
        # Predict
        y_pred_patient = model.predict(X_patient)
        
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
        plt.title(f"{patient} - PLS-DA (Acc: {accuracy:.3f}, MAE: {mae:.3f})", fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, 'patient_specific', f'{patient}_predictions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("=" * 70)
    print("CAPS Score Classification - PLS-DA (Partial Least Squares Discriminant Analysis)")
    print("=" * 70)
    
    # Get outputs directory
    outputs_dir = get_outputs_dir()
    print(f"\nOutputs will be saved to: {outputs_dir}")
    
    # Load data
    combined_df, patient_info = load_all_patient_data()
    print(f"\nCombined dataset: {combined_df.shape[0]} samples from {len(patient_info)} patients")
    
    # Prepare features
    X, y = prepare_features(combined_df)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of CAPS score classes: {len(np.unique(y))}")
    
    # Determine optimal number of components (use 2 or min of features, classes-1)
    n_components = min(2, X.shape[1], len(np.unique(y)) - 1)
    if n_components < 1:
        n_components = 1
    
    # Train PLS-DA model
    results = train_plsda_model(X, y, n_components=n_components)
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("Generating Visualizations")
    print(f"{'='*70}")
    
    plot_latent_variables(results, outputs_dir)
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
    for i, feat in enumerate(results['feature_names'], 1):
        print(f"  {i}. {feat}")
    print(f"\nPLS-DA Model Performance:")
    print(f"  Number of Components: {results['n_components']}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"\nAll outputs saved to: {outputs_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

