"""
This module contains the implementation of a logistic regression model for predicting LFP data from TRPTSD data.
This module is part of the TRPTSD project for the Suthana Lab Group.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# Data Loading and Preprocessing
def load_data(trptsd_path, lfp_path):
    """
    Load TRPTSD and LFP data from CSV files.
    
    Parameters:
    trptsd_path (str): Path to the TRPTSD data CSV file.
    lfp_path (str): Path to the LFP data CSV file.
    
    Returns:
    pd.DataFrame, pd.DataFrame: Loaded TRPTSD and LFP data.
    """
    trptsd_data = pd.read_csv(trptsd_path)
    lfp_data = pd.read_csv(lfp_path)
    return trptsd_data, lfp_data

def preprocess_data(trptsd_data, lfp_data):
    """
    Preprocess the TRPTSD and LFP data for modeling.    
    Parameters:
    trptsd_data (pd.DataFrame): TRPTSD data.
    lfp_data (pd.DataFrame): LFP data.
    Returns:
    pd.DataFrame, pd.Series: Features and target variable for modeling.
    """
    # Example preprocessing: Align data by timestamp and handle missing values
    merged_data = pd.merge(trptsd_data, lfp_data, on='timestamp')
    merged_data = merged_data.dropna()
    X = merged_data.drop(columns=['timestamp', 'lfp_signal'])
    y = merged_data['lfp_signal']
    return X, y

# Model Training and Evaluation
def train_model(X, y):
    """
    Train a logistic regression model.
    Parameters:
    X (pd.DataFrame): Features for training.
    y (pd.Series): Target variable for training.
    Returns:
    LogisticRegression, pd.DataFrame, pd.Series: Trained model, X_test, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model, X_test, y_test

def predict(model, X_new):
    """
    Make predictions using the trained model.
    Parameters:
    model (LogisticRegression): Trained logistic regression model.
    X_new (pd.DataFrame): New data for prediction.
    Returns:
    np.ndarray: Predicted values.
    """
    return model.predict(X_new)

def plot_roc_curve(model, X_test, y_test, title="ROC Curve", save_path=None, show_plot=True):
    """
    Plot the ROC curve for a trained binary classifier.

    Parameters:
    model (LogisticRegression): Trained logistic regression model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series or np.ndarray): True binary labels for the test set.
    title (str): Plot title.
    save_path (str or None): If provided, saves the figure to this path.
    show_plot (bool): Whether to display the plot with plt.show().
    """
    # Get predicted probabilities for the positive class
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback to decision_function if predict_proba is not available
        y_scores = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_value = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_value:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()

# Example usage
if __name__ == "__main__":  
    trptsd_path = 'data/trptsd_data.csv'
    lfp_path = 'data/lfp_data.csv'    
    trptsd_data, lfp_data = load_data(trptsd_path, lfp_path)
    X, y = preprocess_data(trptsd_data, lfp_data)
    model, X_test, y_test = train_model(X, y)
    # Example prediction on new data
    X_new = X.sample(5) # Replace with actual new data
    predictions = predict(model, X_new)
    print("Predictions:", predictions)
    # Plot ROC curve using the held-out test set
    plot_roc_curve(
        model,
        X_test,
        y_test,
        title="Logistic Regression ROC",
        save_path=None,  # e.g., "outputs/logistic_regression_roc.png"
        show_plot=True
    )
