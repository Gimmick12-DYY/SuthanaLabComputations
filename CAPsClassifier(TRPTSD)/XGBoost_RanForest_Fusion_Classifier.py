"""
Classifier Model for CAPS score using stacked soft tree ensemble (RandomForest + XGBoost-like)

This script now adapts to your data schema. It will:
- Load the CSV (default: CosinorRegressionModel(TRPTSD)/data/RNS_G_Full_output.csv)
- Auto-detect the CAPS target column
- Build features from available columns:
  * All numeric columns except the target and obvious index/time columns
  * Optionally: include cosinor/linearAR/entropy columns by name pattern if present
  * Time features derived from Region start time (hour, dow, sin/cos) if present
  * One-hot encode categorical columns like Label
- Train soft RandomForest and soft XGBoost-like models and a stacking meta-model
- Evaluate with MSE and plots
"""

from __future__ import annotations
import os
import math
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Data Loading/Prep Helpers
# -------------------------

def _detect_target_column(cols: List[str], provided: Optional[str] = None) -> str:
    if provided and provided in cols:
        return provided
    candidates = [
        "CAPS_score", "CAPS", "caps", "CAPSScore", "CAPS_total", "CAPS Total",
        "caps_score", "caps_total",
    ]
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(
        f"Could not find CAPS target column. Looked for {candidates}. Columns available: {cols}"
    )


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "Region start time" in df.columns:
        ts = pd.to_datetime(df["Region start time"], errors="coerce")
        df = df.copy()
        df["hour"] = ts.dt.hour
        df["dow"] = ts.dt.dayofweek
        # Cyclic encoding for hour of day
        df["sin_hour"] = np.sin(2 * math.pi * df["hour"] / 24.0)
        df["cos_hour"] = np.cos(2 * math.pi * df["hour"] / 24.0)
    return df


def _select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    include_patterns: Optional[List[str]] = None,
) -> List[str]:
    # Exclude obvious non-feature columns
    exclude = {target_col, "Region start time", "date", "Unnamed: 0"}

    cols = []
    if include_patterns:
        pats = [p.lower() for p in include_patterns]
        for c in df.columns:
            if c in exclude:
                continue
            cl = c.lower()
            if any(cl.startswith(p) or (p in cl) for p in pats):
                cols.append(c)
    else:
        # Default: all numeric columns except target and excluded
        for c in df.select_dtypes(include=[np.number]).columns:
            if c not in exclude:
                cols.append(c)
        # If we ended with no numeric features (unlikely), fall back to anything except excluded
        if not cols:
            cols = [c for c in df.columns if c not in exclude]

    # Ensure we actually have features
    if not cols:
        raise ValueError("No feature columns selected. Check your include_patterns or data.")
    return cols


def load_caps_dataset(
    csv_path: str,
    target_col: Optional[str] = None,
    include_patterns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)

    # Drop obvious index-like columns
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])  # auto index column from some exports

    # Add time-derived features if timestamp exists
    df = _add_time_features(df)

    # Detect target
    tcol = _detect_target_column(df.columns.tolist(), provided=target_col)

    # Build base feature set by pattern or numeric default
    base_cols = _select_feature_columns(df, tcol, include_patterns=include_patterns)
    base_df = df[base_cols].copy()

    # Include categorical columns (object or category) by one-hot, except target
    cat_cols = [c for c in base_df.columns if base_df[c].dtype == object or str(base_df[c].dtype).startswith("category")]
    if cat_cols:
        base_df = pd.get_dummies(base_df, columns=cat_cols, drop_first=True)

    # Impute missing values with median per column
    for c in base_df.columns:
        if base_df[c].isna().any():
            base_df[c] = base_df[c].fillna(base_df[c].median())

    # Target values
    y = df[tcol].values.astype(np.float32)
    # Remove rows with nan targets
    mask = ~np.isnan(y)
    X = base_df.values[mask].astype(np.float32)
    y = y[mask]

    # y should be 2D for torch regression
    y = y.reshape(-1, 1)

    feature_names = base_df.columns.tolist()
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"Empty X after preprocessing: X={X.shape}, y={y.shape}")

    return X, y, feature_names


def train_test_split_array(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_test = max(1, int(round(test_size * n)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# -------------------
# Soft Tree Ensembles
# -------------------

class SoftDecisionTree(nn.Module):
    """Soft decision tree with logistic gates over n_features.
    Note: This is a differentiable surrogate; it is not the exact RF/XGBoost algorithm.
    """
    def __init__(self, n_features: int, depth: int = 3):
        super().__init__()
        self.depth = depth
        self.n_features = n_features
        self.n_leaves = 2 ** depth
        self.n_internal_nodes = 2 ** depth - 1
        # One linear gate per internal node over all features
        self.decision_nodes = nn.ModuleList([nn.Linear(n_features, 1) for _ in range(self.n_internal_nodes)])
        # Leaf values
        self.leaf_values = nn.Parameter(torch.randn(self.n_leaves, 1) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        # Compute gate probabilities at all internal nodes
        gates = []
        for node in self.decision_nodes:
            p = torch.sigmoid(node(x))  # (batch, 1)
            gates.append(p)
        gates = torch.stack(gates, dim=1)  # (batch, n_internal, 1)

        # Compute probability of each leaf for each sample
        leaf_probs = []
        for leaf in range(self.n_leaves):
            # Binary path bits from root to this leaf
            path = []
            idx = leaf
            for _ in range(self.depth):
                path.append(idx % 2)
                idx //= 2
            path = path[::-1]

            prob = torch.ones(batch, 1, device=x.device)
            node_index = 0
            for decision in path:
                p = gates[:, node_index]
                if decision == 0:
                    prob = prob * (1 - p)
                else:
                    prob = prob * p
                node_index += 1
            leaf_probs.append(prob)
        leaf_probs = torch.cat(leaf_probs, dim=1)  # (batch, n_leaves)
        out = torch.matmul(leaf_probs, self.leaf_values)  # (batch, 1)
        return out


class SoftRandomForest(nn.Module):
    """Ensemble of soft decision trees; outputs the mean prediction."""
    def __init__(self, n_features: int, n_trees: int = 10, tree_depth: int = 3):
        super().__init__()
        self.trees = nn.ModuleList([SoftDecisionTree(n_features, depth=tree_depth) for _ in range(n_trees)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = [t(x) for t in self.trees]
        preds = torch.stack(preds, dim=0)  # (n_trees, batch, 1)
        return preds.mean(dim=0)


class SoftXGBoost(nn.Module):
    """Additive ensemble of soft decision trees; simplistic gradient boosting surrogate.
    Note: This is a naive additive model; for production, use real XGBoost.
    """
    def __init__(self, n_features: int, n_estimators: int = 10, tree_depth: int = 3, learning_rate: float = 0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.trees = nn.ModuleList([SoftDecisionTree(n_features, depth=tree_depth) for _ in range(n_estimators)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.size(0), 1, device=x.device)
        for t in self.trees:
            out = out + self.learning_rate * t(x)
        return out


class StackingFusionModel(nn.Module):
    """Simple meta-model over RF and XGB predictions."""
    def __init__(self, rf_model: nn.Module, xgb_model: nn.Module):
        super().__init__()
        self.rf = rf_model
        self.xgb = xgb_model
        self.meta = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rf_pred = self.rf(x)
        xgb_pred = self.xgb(x)
        fused_in = torch.cat([rf_pred, xgb_pred], dim=1)
        return self.meta(fused_in)

# ---------------------------
# Training / Evaluation Utils
# ---------------------------

def train_model(model: nn.Module, optimizer, criterion, x, y, num_epochs: int = 300):
    model.train()
    hist = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        hist.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - loss: {loss.item():.4f}")
    return hist


def evaluate_model(model: nn.Module, x, y):
    model.eval()
    with torch.no_grad():
        preds = model(x)
        mse = nn.MSELoss()(preds, y).item()
        y_np = y.detach().cpu().numpy().reshape(-1)
        p_np = preds.detach().cpu().numpy().reshape(-1)
        # R^2
        ss_res = float(np.sum((y_np - p_np) ** 2))
        ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return p_np, mse, r2

# -----
# Main
# -----

def main(
    csv_path: Optional[str] = None,
    target_col: Optional[str] = None,
    include_patterns: Optional[List[str]] = None,
    n_trees: int = 20,
    n_estimators: int = 30,
    tree_depth: int = 3,
    epochs: int = 300,
    lr_base: float = 0.01,
):
    if csv_path is None:
        # Default to repo-relative sample path
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.normpath(os.path.join(here, "..", "CosinorRegressionModel(TRPTSD)", "data", "RNS_G_Full_output.csv"))

    # Default feature patterns: target cosinor/linearAR/entropy and a couple of known columns if present
    if include_patterns is None:
        include_patterns = [
            "cosinor", "linearar", "linar", "entropy",
            "pattern a channel 2", "episode starts with rx", "hour", "sin_hour", "cos_hour", "dow",
        ]

    print(f"Loading dataset: {csv_path}")
    X, y, feat_names = load_caps_dataset(csv_path, target_col=target_col, include_patterns=include_patterns)
    print(f"Data shape: X={X.shape}, y={y.shape}, features={len(feat_names)}")

    X_train, X_test, y_train, y_test = train_test_split_array(X, y, test_size=0.2, seed=SEED)

    # Tensors
    xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(y_test, dtype=torch.float32, device=device)

    n_features = X_train.shape[1]

    rf = SoftRandomForest(n_features=n_features, n_trees=n_trees, tree_depth=tree_depth).to(device)
    xgb = SoftXGBoost(n_features=n_features, n_estimators=n_estimators, tree_depth=tree_depth, learning_rate=0.1).to(device)

    crit = nn.MSELoss()
    opt_rf = optim.Adam(rf.parameters(), lr=lr_base)
    opt_xgb = optim.Adam(xgb.parameters(), lr=lr_base)

    print("Training Soft RandomForest...")
    loss_rf = train_model(rf, opt_rf, crit, xtr, ytr, num_epochs=epochs)

    print("Training Soft XGBoost-like...")
    loss_xgb = train_model(xgb, opt_xgb, crit, xtr, ytr, num_epochs=epochs)

    # Evaluate base models
    rf_preds, rf_mse, rf_r2 = evaluate_model(rf, xte, yte)
    xgb_preds, xgb_mse, xgb_r2 = evaluate_model(xgb, xte, yte)
    print(f"RF Test MSE: {rf_mse:.4f} | R2: {rf_r2:.4f}")
    print(f"XGB Test MSE: {xgb_mse:.4f} | R2: {xgb_r2:.4f}")

    # Stacking
    stack = StackingFusionModel(rf, xgb).to(device)
    opt_stack = optim.Adam(stack.parameters(), lr=lr_base)
    print("Training Stacking Fusion Model...")
    loss_stack = train_model(stack, opt_stack, crit, xtr, ytr, num_epochs=epochs)

    stack_preds, stack_mse, stack_r2 = evaluate_model(stack, xte, yte)
    print(f"Stack Test MSE: {stack_mse:.4f} | R2: {stack_r2:.4f}")

    # ------------
    # Visualizations
    # ------------
    plt.figure(figsize=(16, 12))

    # 1) Training loss curves
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(loss_rf, label="RF Loss", color="red")
    ax1.plot(loss_xgb, label="XGB Loss", color="blue")
    ax1.plot(loss_stack, label="Stack Loss", color="green")
    ax1.set_title("Training Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True)
    ax1.legend()

    # 2) True vs Predicted (Stack)
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(y_test.reshape(-1), rf_preds, alpha=0.6, label="RF", color="red", s=25)
    ax2.scatter(y_test.reshape(-1), xgb_preds, alpha=0.6, label="XGB", color="blue", s=25)
    ax2.scatter(y_test.reshape(-1), stack_preds, alpha=0.6, label="Stack", color="green", s=25)
    y_min = float(min(y_test.min(), rf_preds.min(), xgb_preds.min(), stack_preds.min()))
    y_max = float(max(y_test.max(), rf_preds.max(), xgb_preds.max(), stack_preds.max()))
    ax2.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1)
    ax2.set_title("True vs Predicted (Test)")
    ax2.set_xlabel("True CAPS")
    ax2.set_ylabel("Predicted CAPS")
    ax2.grid(True)
    ax2.legend()

    # 3) Residuals histogram (Stack)
    ax3 = plt.subplot(2, 2, 3)
    resid_stack = y_test.reshape(-1) - stack_preds
    ax3.hist(resid_stack, bins=20, color="purple", alpha=0.7)
    ax3.set_title("Stack Residuals Distribution")
    ax3.set_xlabel("Error (True - Pred)")
    ax3.set_ylabel("Count")
    ax3.grid(True)

    # 4) Absolute error curves (sorted by pred)
    ax4 = plt.subplot(2, 2, 4)
    idx_sort = np.argsort(stack_preds)
    ax4.plot(np.abs(y_test.reshape(-1)[idx_sort] - rf_preds[idx_sort]), "r--", label="RF |AE|")
    ax4.plot(np.abs(y_test.reshape(-1)[idx_sort] - xgb_preds[idx_sort]), "b--", label="XGB |AE|")
    ax4.plot(np.abs(y_test.reshape(-1)[idx_sort] - stack_preds[idx_sort]), "g--", label="Stack |AE|")
    ax4.set_title("Absolute Error (sorted by Stack pred)")
    ax4.set_xlabel("Sample Index (sorted)")
    ax4.set_ylabel("|Error|")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()

    # Save figures
    try:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, "classifier_analysis.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")

        fig = plt.gcf()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        renderer = fig.canvas.get_renderer()
        for i, ax in enumerate(fig.axes, start=1):
            try:
                bbox = ax.get_tightbbox(renderer)
                bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
                sub_path = os.path.join(out_dir, f"classifier_analysis_subplot{i}.png")
                fig.savefig(sub_path, dpi=300, bbox_inches=bbox_inches)
            except Exception as e:
                print(f"Warning: could not save subplot {i}: {e}")
        print(f"Saved plots to: {out_dir}")
    except Exception as e:
        print(f"Warning: could not save figures: {e}")

    plt.show()


if __name__ == "__main__":
    # You can override csv_path/target_col via CLI by editing here or passing via environment
    main()
