"""
-------------------------------------------------------------------------------------------------------------------
Cosinor Regression Analysis with 24-Hour Period - Multi-day Fit (per file)

Created: Sept 2025

This script performs a continuous multi-day cosinor fit for each CSV file in a folder.
It infers the correct Pattern Channel column as y, constructs x as hours-since-start,
fits a 24h cosinor via linear regression, and saves per-file plots and spreadsheets.
-------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import seaborn as sns

np.seterr(divide='ignore')


# --- Utilities ---
def parse_dataset_info(file_path):
    name = Path(file_path).name
    parts = name.split('_')
    patient = parts[1] if len(parts) > 1 else 'Unknown'
    pattern_chan = parts[2] if len(parts) > 2 else 'A2'
    pattern_letter = pattern_chan[0]
    channel_num = ''.join(ch for ch in pattern_chan[1:] if ch.isdigit()) or '2'
    key_name = f"{patient}_{pattern_letter}{channel_num}"
    y_col = f"Pattern {pattern_letter} Channel {channel_num}"
    return key_name, y_col


def load_and_prepare_multiday(dataset_path, y_column):
    df = pd.read_csv(dataset_path)
    df['Region start time'] = pd.to_datetime(df['Region start time'])
    df = df.sort_values('Region start time')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df['date'] = df['Region start time'].dt.date
    df['hour'] = df['Region start time'].dt.hour
    t0 = df['Region start time'].iloc[0]
    df['x_hours'] = (df['Region start time'] - t0).dt.total_seconds() / 3600.0
    if y_column not in df.columns:
        raise KeyError(f"Expected y column '{y_column}' not found in {dataset_path}")
    df['y'] = df[y_column]
    # Normalize labels for coloring
    if 'Label' in df.columns:
        pass
    elif 'label' in df.columns:
        df.rename(columns={'label': 'Label'}, inplace=True)
    else:
        df['Label'] = df['Region start time'].dt.strftime('%Y-%m')
    return df


def fit_cosinor_24h(x_hours: np.ndarray, y: np.ndarray):
    """
    Fit classical cosinor: y = M + A*cos(w x) + B*sin(w x), w=2π/24.
    Returns dict with mesor, amplitude, acrophase_hours, r_squared, betas, covariance.
    """
    w = 2 * np.pi / 24.0
    X = np.column_stack([
        np.ones_like(x_hours),
        np.cos(w * x_hours),
        np.sin(w * x_hours),
    ])
    # OLS
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    mesor = beta[0]
    A = beta[1]
    B = beta[2]
    amplitude = np.sqrt(A * A + B * B)
    # acrophase phi such that y = M + amplitude*cos(w x - phi)
    # Mapping: A = amplitude*cos(phi), B = amplitude*sin(phi)
    phi = np.arctan2(B, A)  # radians
    acrophase_hours = (phi / w) % 24.0

    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) if len(y) > 1 else 0.0
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # Covariance estimate
    n = X.shape[0]
    p = X.shape[1]
    sigma2 = ss_res / max(n - p, 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov_beta = sigma2 * XtX_inv

    return {
        'mesor': mesor,
        'amplitude': amplitude,
        'acrophase_hours': acrophase_hours,
        'r_squared': r_squared,
        'beta': beta,
        'cov_beta': cov_beta,
        'y_hat': y_hat,
        'ss_res': ss_res,
        'ss_tot': ss_tot,
    }


def plot_multiday(df: pd.DataFrame, y_hat: np.ndarray, key_name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Observed colored by month label, fitted as a black line
    plt.figure(figsize=(14, 6))
    label_col = 'Label'
    unique_labels = [lab for lab in df[label_col].dropna().unique()]
    palette = sns.color_palette('tab20', n_colors=max(4, len(unique_labels) or 4))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}
    for lab in unique_labels:
        sub = df[df[label_col] == lab]
        plt.scatter(sub['Region start time'], sub['y'], s=12, label=str(lab), color=color_map[lab], alpha=0.7)
    if not unique_labels:
        plt.scatter(df['Region start time'], df['y'], s=12, alpha=0.7)
    # Fitted curve (downsample to avoid overplotting thick band)
    max_points = 1500
    step = max(1, int(np.ceil(len(y_hat) / max_points)))
    plt.plot(df['Region start time'][::step], y_hat[::step], color='black', linewidth=1, alpha=0.85, label='Fitted (24h)')
    plt.title(f'Observed vs Fitted (24h Cosinor) - {key_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    if unique_labels:
        plt.legend(ncol=4, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{key_name}_observed_vs_fitted.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Residuals histogram
    plt.figure(figsize=(8, 5))
    residuals = df['y'] - y_hat
    plt.hist(residuals, bins=30, alpha=0.85)
    plt.title(f'Residuals - {key_name}', fontweight='bold')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{key_name}_residuals_hist.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Multi-day cosinor per file: iterate all CSVs in a folder, fit once per file,
    save parameters CSV, enhanced data with fitted values, and plots.
    """
    data_path = "CosinorRegressionModel(TRPTSD)/data/Data_Cosinor_09.15.25"

    base_outputs = Path('CosinorRegressionModel(TRPTSD)/outputs/outputs_multiday')
    base_outputs.mkdir(parents=True, exist_ok=True)

    folder_dir = Path(data_path)
    dataset_list = sorted([str(fp) for fp in folder_dir.glob("RNS_*_Full*.csv")])
    if not dataset_list:
        dataset_list = sorted([str(fp) for fp in folder_dir.glob("*.csv")])
    if not dataset_list:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    print(f"Found {len(dataset_list)} datasets to process (multi-day fit).")

    all_params = []

    for dataset in dataset_list:
        key_name, y_col = parse_dataset_info(dataset)
        print(f"\nProcessing {Path(dataset).name} -> key '{key_name}', y '{y_col}'")

        df = load_and_prepare_multiday(dataset, y_col)
        results = fit_cosinor_24h(df['x_hours'].values.astype(float), df['y'].values.astype(float))

        # Save enhanced data with fitted and residuals
        df_out = df.copy()
        df_out['y_hat'] = results['y_hat']
        df_out['residual'] = df_out['y'] - df_out['y_hat']

        out_dir = base_outputs / key_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Parameters CSV
        params = {
            'Key': key_name,
            'Mesor': results['mesor'],
            'Amplitude': results['amplitude'],
            'Acrophase_hours': results['acrophase_hours'],
            'R_squared': results['r_squared'],
            'N': len(df_out)
        }
        all_params.append(params)
        pd.DataFrame([params]).to_csv(out_dir / f'{key_name}_cosinor_params.csv', index=False)

        # Enhanced data CSV
        df_out.to_csv(out_dir / f'{key_name}_enhanced_with_fit.csv', index=False)

        # Plots
        plots_dir = out_dir / 'plots'
        plot_multiday(df_out, results['y_hat'], key_name, str(plots_dir))

        print(f"  Saved parameters, enhanced data, and plots to {out_dir}")

    # Combined parameters across files
    if all_params:
        params_df = pd.DataFrame(all_params)
        params_df.to_csv(base_outputs / 'all_files_cosinor_params.csv', index=False)
        print(f"\nSaved combined parameters: {base_outputs / 'all_files_cosinor_params.csv'}")


if __name__ == "__main__":
    main()


