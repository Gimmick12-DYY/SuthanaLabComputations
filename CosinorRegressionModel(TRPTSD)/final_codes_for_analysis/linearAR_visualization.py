"""
-------------------------------------------------------------------------------------------------------------------
LinearAR Visualization - Replicate-style plots

Created: November 2025

This script generates replicate-style plots similar to the cosinor analysis for LinearAR metrics.
Uses existing linearAR_Predicted and linearAR_Fit_Residual columns from processed data.
-------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
from matplotlib import gridspec

np.seterr(divide='ignore')


# --- Utilities ---
def parse_dataset_info(file_path):
    """Parse patient and channel info from filename."""
    name = Path(file_path).name
    parts = name.split('_')
    # Format: processed_RNS_A_B2_Complete.csv
    patient = parts[2] if len(parts) > 2 else 'Unknown'
    pattern_chan = parts[3] if len(parts) > 3 else 'B2'
    key_name = f"{patient}_{pattern_chan}"
    return key_name


def load_processed_data(dataset_path):
    """Load and prepare processed data."""
    df = pd.read_csv(dataset_path)
    df['Region start time'] = pd.to_datetime(df['Region start time'])
    df = df.sort_values('Region start time')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Ensure date column
    if 'date' not in df.columns:
        df['date'] = df['Region start time'].dt.date
    else:
        df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Ensure Label column
    if 'Label' not in df.columns:
        df['Label'] = df['Region start time'].dt.strftime('%Y-%m')
    
    return df


def compute_daily_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily statistics for LinearAR residuals.
    Returns DataFrame with columns: date, Label, residual_mean, residual_std, abs_residual_mean, day_index, date_dt
    """
    rows = []
    
    for date, group in df.groupby('date'):
        if len(group) < 1:
            continue
        
        if 'linearAR_Fit_Residual' not in group.columns:
            continue
        
        residuals = group['linearAR_Fit_Residual'].dropna()
        if len(residuals) > 0:
            label_val = group['Label'].iloc[0] if 'Label' in group.columns else None
            rows.append({
                'date': date,
                'Label': label_val,
                'residual_mean': residuals.mean(),
                'residual_std': residuals.std(),
                'abs_residual_mean': residuals.abs().mean(),
            })
    
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return rdf
    
    rdf['date_dt'] = pd.to_datetime(rdf['date'])
    rdf = rdf.sort_values('date_dt').reset_index(drop=True)
    rdf['day_index'] = np.arange(len(rdf))
    return rdf


def plot_linearAR_observed_vs_fitted(df: pd.DataFrame, key_name: str, out_dir: str):
    """Plot LinearAR observed vs fitted (predicted) over time."""
    os.makedirs(out_dir, exist_ok=True)
    
    # Find the pattern column
    pattern_cols = [col for col in df.columns if col.startswith('Pattern ') and 'Channel' in col]
    y_col = None
    for col in pattern_cols:
        if df[col].sum() > 0:  # Find non-zero column
            y_col = col
            break
    
    if y_col is None or 'linearAR_Predicted' not in df.columns:
        print(f"  Warning: Could not find appropriate columns for {key_name}")
        return
    
    plt.figure(figsize=(14, 6))
    
    # Scatter observed values colored by label
    label_col = 'Label'
    unique_labels = [lab for lab in df[label_col].dropna().unique()]
    palette = sns.color_palette('tab20', n_colors=max(4, len(unique_labels) or 4))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}
    
    for lab in unique_labels:
        sub = df[df[label_col] == lab]
        plt.scatter(sub['Region start time'], sub[y_col], s=12, label=str(lab), 
                   color=color_map[lab], alpha=0.7)
    
    # Fitted curve (downsample to avoid overplotting)
    max_points = 1500
    step = max(1, int(np.ceil(len(df) / max_points)))
    plt.plot(df['Region start time'][::step], df['linearAR_Predicted'][::step], 
            color='black', linewidth=1, alpha=0.85, label='Fitted (LinearAR)')
    
    plt.title(f'Observed vs Fitted (LinearAR) - {key_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    if unique_labels:
        plt.legend(ncol=4, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{key_name}_linearAR_observed_vs_fitted.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals_histogram(df: pd.DataFrame, key_name: str, out_dir: str):
    """Plot residuals histogram."""
    os.makedirs(out_dir, exist_ok=True)
    
    if 'linearAR_Fit_Residual' not in df.columns:
        return
    
    plt.figure(figsize=(8, 5))
    residuals = df['linearAR_Fit_Residual'].dropna()
    plt.hist(residuals, bins=30, alpha=0.85)
    plt.title(f'LinearAR Residuals - {key_name}', fontweight='bold')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{key_name}_linearAR_residuals_hist.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals_replicate(rdf: pd.DataFrame, key_name: str, out_dir: str):
    """
    Replicate-style plot: left scatter of daily mean absolute residuals over time with smoothed lines,
    right violin comparing pre vs post phases.
    """
    if rdf.empty:
        return
    os.makedirs(out_dir, exist_ok=True)
    
    # Prefer semantic split: pre (labels starting with 'B') vs post (starting with 'M')
    has_labels = 'Label' in rdf.columns and rdf['Label'].notna().any()
    split_index = None
    
    if has_labels:
        pre_mask = rdf['Label'].astype(str).str.startswith('B')
        post_mask = rdf['Label'].astype(str).str.startswith('M')
        left_df = rdf[pre_mask].copy()
        right_df = rdf[post_mask].copy()
        # For vertical guideline, pick first index of post if exists
        if post_mask.any():
            split_index = int(rdf[post_mask].iloc[0]['day_index'])
        else:
            split_index = len(rdf) // 2
    else:
        split_index = len(rdf) // 2
        left_df = rdf.iloc[:split_index].copy()
        right_df = rdf.iloc[split_index:].copy()
    
    # Smoothing via rolling mean
    def smooth(y, window):
        if len(y) < 3:
            return y
        w = max(3, window if window % 2 == 1 else window + 1)
        return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().values
    
    left_smooth = smooth(left_df['abs_residual_mean'].values, window=max(5, len(left_df)//10 or 5))
    right_smooth = smooth(right_df['abs_residual_mean'].values, window=max(5, len(right_df)//10 or 5))
    
    # Figure layout
    plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.25)
    ax_main = plt.subplot(gs[0])
    ax_vio = plt.subplot(gs[1])
    
    # Colors
    col_left = '#f1c40f'  # yellow (Pre, B*)
    col_right = '#2e86de'  # blue (Post, M*)
    
    # Scatter colored by month label
    if has_labels:
        labs = rdf['Label'].astype(str).unique().tolist()
        pal = sns.color_palette('tab20', n_colors=max(4, len(labs)))
        cmap = {lab: pal[i % len(pal)] for i, lab in enumerate(labs)}
        for lab in labs:
            sub = rdf[rdf['Label'] == lab]
            ax_main.scatter(sub['day_index'], sub['abs_residual_mean'], s=16, color=cmap[lab], 
                          alpha=0.7, label=str(lab))
        ax_main.legend(ncol=6, fontsize=7, frameon=False, loc='upper left', 
                      bbox_to_anchor=(0, 1.18))
    else:
        ax_main.scatter(rdf['day_index'], rdf['abs_residual_mean'], s=16, alpha=0.7, color='gray')
    
    # Smoothed lines for pre/post
    ax_main.plot(left_df['day_index'], left_smooth, color=col_left, linewidth=2, 
                label='Pre (B*)')
    ax_main.plot(right_df['day_index'], right_smooth, color=col_right, linewidth=2, 
                label='Post (M*)')
    
    # Vertical split line
    if split_index is not None:
        ax_main.axvline(split_index, color='#e84393', linestyle='--', linewidth=2)
    
    ax_main.set_title(f'LinearAR Residuals (Abs Mean) over Time - {key_name}', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('Day Index')
    ax_main.set_ylabel('Mean Absolute Residual')
    ax_main.grid(True, alpha=0.3)
    
    # Violin plot comparing halves
    data_vio = [left_df['abs_residual_mean'].dropna().values, right_df['abs_residual_mean'].dropna().values]
    parts = ax_vio.violinplot(data_vio, positions=[0, 1], showmeans=False, 
                             showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(col_left if i == 0 else col_right)
        pc.set_alpha(0.8)
    ax_vio.set_xticks([0, 1])
    ax_vio.set_xticklabels(['Pre (B*)', 'Post (M*)'])
    ax_vio.set_ylabel('Mean Absolute Residual')
    ax_vio.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{key_name}_linearAR_residuals_replicate.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Generate LinearAR visualizations for all processed files.
    """
    data_path = "CAPsClassifier(TRPTSD)/data/Processed_Data"
    
    base_outputs = Path('CosinorRegressionModel(TRPTSD)/outputs/outputs_linearAR')
    base_outputs.mkdir(parents=True, exist_ok=True)
    
    folder_dir = Path(data_path)
    dataset_list = sorted([str(fp) for fp in folder_dir.glob("processed_*.csv")])
    
    if not dataset_list:
        raise FileNotFoundError(f"No processed CSV files found in {data_path}")
    
    print(f"Found {len(dataset_list)} datasets to process.")
    
    all_stats = []
    
    for dataset in dataset_list:
        key_name = parse_dataset_info(dataset)
        print(f"\nProcessing {Path(dataset).name} -> key '{key_name}'")
        
        df = load_processed_data(dataset)
        
        out_dir = base_outputs / key_name
        out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = out_dir / 'plots'
        
        # --- LinearAR Analysis ---
        print(f"  Generating LinearAR plots...")
        plot_linearAR_observed_vs_fitted(df, key_name, str(plots_dir))
        plot_residuals_histogram(df, key_name, str(plots_dir))
        
        rdf = compute_daily_residuals(df)
        if not rdf.empty:
            # Save daily residuals data
            rdf.to_csv(out_dir / f'{key_name}_linearAR_daily_residuals.csv', index=False)
            # Plot residuals over time
            plot_residuals_replicate(rdf, key_name, str(plots_dir))
            
            # Compute statistics
            has_labels = 'Label' in rdf.columns and rdf['Label'].notna().any()
            if has_labels:
                pre_mask = rdf['Label'].astype(str).str.startswith('B')
                post_mask = rdf['Label'].astype(str).str.startswith('M')
                pre_res = rdf[pre_mask]['abs_residual_mean'].mean() if pre_mask.any() else np.nan
                post_res = rdf[post_mask]['abs_residual_mean'].mean() if post_mask.any() else np.nan
            else:
                mid = len(rdf) // 2
                pre_res = rdf.iloc[:mid]['abs_residual_mean'].mean()
                post_res = rdf.iloc[mid:]['abs_residual_mean'].mean()
            
            stats = {
                'Key': key_name,
                'LinearAR_Overall_Abs_Residual': rdf['abs_residual_mean'].mean(),
                'LinearAR_Pre_Abs_Residual': pre_res,
                'LinearAR_Post_Abs_Residual': post_res,
            }
        else:
            stats = {'Key': key_name}
        
        all_stats.append(stats)
        print(f"  Saved plots and data to {out_dir}")
    
    # Combined statistics across files
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(base_outputs / 'all_files_linearAR_stats.csv', index=False)
        print(f"\nSaved combined statistics: {base_outputs / 'all_files_linearAR_stats.csv'}")
    
    print("\n" + "="*70)
    print("Processing complete!")
    print(f"All outputs saved to: {base_outputs}")
    print("="*70)


if __name__ == "__main__":
    main()

