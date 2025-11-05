"""
-------------------------------------------------------------------------------------------------------------------
LinearAR and Sample Entropy Visualization - Replicate-style plots

Created: November 2025

This script generates replicate-style plots similar to the cosinor analysis for:
1. LinearAR metrics (observed vs predicted, daily R²)
2. Sample Entropy metrics (daily values over time)

Generates plots matching the style of cosinor_analysis_24h_multiday.py
-------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
from matplotlib import gridspec
from sklearn.metrics import r2_score

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


def compute_daily_r2_linearAR(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily R² for LinearAR by comparing observed vs predicted values.
    Returns DataFrame with columns: date, Label, r2, r2_pct, day_index, date_dt
    """
    rows = []
    
    # Determine which column has the observed values
    # Looking for Pattern columns (A or B, Channel 1 or 2)
    pattern_cols = [col for col in df.columns if col.startswith('Pattern ') and 'Channel' in col]
    
    for date, group in df.groupby('date'):
        if len(group) < 6:
            continue
        
        # Get predicted values
        if 'linearAR_Predicted' not in group.columns:
            continue
            
        y_pred = group['linearAR_Predicted'].values
        
        # Find the actual pattern channel being analyzed (non-zero values)
        best_r2 = -np.inf
        for col in pattern_cols:
            if col in group.columns:
                y_obs = group[col].values
                # Only calculate if there's variation
                if len(y_obs) > 0 and y_obs.std() > 0:
                    try:
                        r2 = r2_score(y_obs, y_pred)
                        if r2 > best_r2:
                            best_r2 = r2
                    except:
                        continue
        
        if best_r2 > -np.inf:
            label_val = group['Label'].iloc[0] if 'Label' in group.columns else None
            rows.append({
                'date': date,
                'Label': label_val,
                'r2': best_r2,
                'r2_pct': best_r2 * 100.0 if pd.notna(best_r2) else np.nan,
            })
    
    r2df = pd.DataFrame(rows)
    if r2df.empty:
        return r2df
    
    r2df['date_dt'] = pd.to_datetime(r2df['date'])
    r2df = r2df.sort_values('date_dt').reset_index(drop=True)
    r2df['day_index'] = np.arange(len(r2df))
    return r2df


def compute_daily_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily average Sample Entropy.
    Returns DataFrame with columns: date, Label, entropy_avg, day_index, date_dt
    """
    rows = []
    
    for date, group in df.groupby('date'):
        if len(group) < 1:
            continue
        
        if 'Sample Entropy' not in group.columns:
            continue
        
        entropy_vals = group['Sample Entropy'].dropna()
        if len(entropy_vals) > 0:
            label_val = group['Label'].iloc[0] if 'Label' in group.columns else None
            rows.append({
                'date': date,
                'Label': label_val,
                'entropy_avg': entropy_vals.mean(),
                'entropy_std': entropy_vals.std(),
            })
    
    edf = pd.DataFrame(rows)
    if edf.empty:
        return edf
    
    edf['date_dt'] = pd.to_datetime(edf['date'])
    edf = edf.sort_values('date_dt').reset_index(drop=True)
    edf['day_index'] = np.arange(len(edf))
    return edf


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
    
    # Residuals histogram
    if 'linearAR_Fit_Residual' in df.columns:
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


def plot_r2_replicate(r2df: pd.DataFrame, key_name: str, out_dir: str, 
                     metric_name: str = "LinearAR R²", y_col: str = 'r2_pct'):
    """
    Replicate-style plot: left scatter over time with smoothed lines,
    right violin comparing pre vs post phases.
    """
    if r2df.empty:
        return
    os.makedirs(out_dir, exist_ok=True)
    
    # Prefer semantic split: pre (labels starting with 'B') vs post (starting with 'M')
    has_labels = 'Label' in r2df.columns and r2df['Label'].notna().any()
    split_index = None
    
    if has_labels:
        pre_mask = r2df['Label'].astype(str).str.startswith('B')
        post_mask = r2df['Label'].astype(str).str.startswith('M')
        left_df = r2df[pre_mask].copy()
        right_df = r2df[post_mask].copy()
        # For vertical guideline, pick first index of post if exists
        if post_mask.any():
            split_index = int(r2df[post_mask].iloc[0]['day_index'])
        else:
            split_index = len(r2df) // 2
    else:
        split_index = len(r2df) // 2
        left_df = r2df.iloc[:split_index].copy()
        right_df = r2df.iloc[split_index:].copy()
    
    # Smoothing via rolling mean
    def smooth(y, window):
        if len(y) < 3:
            return y
        w = max(3, window if window % 2 == 1 else window + 1)
        return pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().values
    
    left_smooth = smooth(left_df[y_col].values, window=max(5, len(left_df)//10 or 5))
    right_smooth = smooth(right_df[y_col].values, window=max(5, len(right_df)//10 or 5))
    
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
        labs = r2df['Label'].astype(str).unique().tolist()
        pal = sns.color_palette('tab20', n_colors=max(4, len(labs)))
        cmap = {lab: pal[i % len(pal)] for i, lab in enumerate(labs)}
        for lab in labs:
            sub = r2df[r2df['Label'] == lab]
            ax_main.scatter(sub['day_index'], sub[y_col], s=16, color=cmap[lab], 
                          alpha=0.7, label=str(lab))
        ax_main.legend(ncol=6, fontsize=7, frameon=False, loc='upper left', 
                      bbox_to_anchor=(0, 1.18))
    else:
        ax_main.scatter(r2df['day_index'], r2df[y_col], s=16, alpha=0.7, color='gray')
    
    # Smoothed lines for pre/post
    ax_main.plot(left_df['day_index'], left_smooth, color=col_left, linewidth=2, 
                label='Pre (B*)')
    ax_main.plot(right_df['day_index'], right_smooth, color=col_right, linewidth=2, 
                label='Post (M*)')
    
    # Vertical split line
    if split_index is not None:
        ax_main.axvline(split_index, color='#e84393', linestyle='--', linewidth=2)
    
    ax_main.set_title(f'{metric_name} over Time - {key_name}', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('Day Index')
    ax_main.set_ylabel(metric_name)
    ax_main.grid(True, alpha=0.3)
    
    # Violin plot comparing halves
    data_vio = [left_df[y_col].dropna().values, right_df[y_col].dropna().values]
    parts = ax_vio.violinplot(data_vio, positions=[0, 1], showmeans=False, 
                             showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(col_left if i == 0 else col_right)
        pc.set_alpha(0.8)
    ax_vio.set_xticks([0, 1])
    ax_vio.set_xticklabels(['Pre (B*)', 'Post (M*)'])
    ax_vio.set_ylabel(metric_name)
    ax_vio.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    filename = f'{key_name}_{metric_name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").lower()}_replicate.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_entropy_timeseries(df: pd.DataFrame, key_name: str, out_dir: str):
    """Plot Sample Entropy over time with trend."""
    os.makedirs(out_dir, exist_ok=True)
    
    if 'Sample Entropy' not in df.columns:
        return
    
    plt.figure(figsize=(14, 6))
    
    # Scatter colored by label
    label_col = 'Label'
    unique_labels = [lab for lab in df[label_col].dropna().unique()]
    palette = sns.color_palette('tab20', n_colors=max(4, len(unique_labels) or 4))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}
    
    for lab in unique_labels:
        sub = df[df[label_col] == lab]
        plt.scatter(sub['Region start time'], sub['Sample Entropy'], s=12, 
                   label=str(lab), color=color_map[lab], alpha=0.7)
    
    # Add rolling average trend
    if len(df) > 100:
        window = max(50, len(df) // 50)
        df_sorted = df.sort_values('Region start time')
        rolling_avg = df_sorted['Sample Entropy'].rolling(window=window, center=True, 
                                                          min_periods=1).mean()
        max_points = 1500
        step = max(1, int(np.ceil(len(rolling_avg) / max_points)))
        plt.plot(df_sorted['Region start time'][::step], rolling_avg[::step], 
                color='black', linewidth=2, alpha=0.85, label='Trend (Rolling Avg)')
    
    plt.title(f'Sample Entropy over Time - {key_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Sample Entropy')
    if unique_labels:
        plt.legend(ncol=4, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{key_name}_entropy_timeseries.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Generate LinearAR and Sample Entropy visualizations for all processed files.
    """
    data_path = "CAPsClassifier(TRPTSD)/data/Processed_Data"
    
    base_outputs = Path('CosinorRegressionModel(TRPTSD)/outputs/outputs_linearAR_entropy')
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
        
        r2df = compute_daily_r2_linearAR(df)
        if not r2df.empty:
            # Save daily R² data
            r2df.to_csv(out_dir / f'{key_name}_linearAR_daily_r2.csv', index=False)
            # Plot R² over time
            plot_r2_replicate(r2df, key_name, str(plots_dir), 
                            metric_name="LinearAR R² (%)", y_col='r2_pct')
            
            # Compute statistics
            has_labels = 'Label' in r2df.columns and r2df['Label'].notna().any()
            if has_labels:
                pre_mask = r2df['Label'].astype(str).str.startswith('B')
                post_mask = r2df['Label'].astype(str).str.startswith('M')
                pre_r2 = r2df[pre_mask]['r2_pct'].mean() if pre_mask.any() else np.nan
                post_r2 = r2df[post_mask]['r2_pct'].mean() if post_mask.any() else np.nan
            else:
                mid = len(r2df) // 2
                pre_r2 = r2df.iloc[:mid]['r2_pct'].mean()
                post_r2 = r2df.iloc[mid:]['r2_pct'].mean()
            
            stats = {
                'Key': key_name,
                'LinearAR_Overall_R2_pct': r2df['r2_pct'].mean(),
                'LinearAR_Pre_R2_pct': pre_r2,
                'LinearAR_Post_R2_pct': post_r2,
            }
        else:
            stats = {'Key': key_name}
        
        # --- Sample Entropy Analysis ---
        print(f"  Generating Sample Entropy plots...")
        plot_entropy_timeseries(df, key_name, str(plots_dir))
        
        edf = compute_daily_entropy(df)
        if not edf.empty:
            # Save daily entropy data
            edf.to_csv(out_dir / f'{key_name}_entropy_daily.csv', index=False)
            # Plot entropy over time
            plot_r2_replicate(edf, key_name, str(plots_dir), 
                            metric_name="Sample Entropy", y_col='entropy_avg')
            
            # Compute statistics
            has_labels = 'Label' in edf.columns and edf['Label'].notna().any()
            if has_labels:
                pre_mask = edf['Label'].astype(str).str.startswith('B')
                post_mask = edf['Label'].astype(str).str.startswith('M')
                pre_ent = edf[pre_mask]['entropy_avg'].mean() if pre_mask.any() else np.nan
                post_ent = edf[post_mask]['entropy_avg'].mean() if post_mask.any() else np.nan
            else:
                mid = len(edf) // 2
                pre_ent = edf.iloc[:mid]['entropy_avg'].mean()
                post_ent = edf.iloc[mid:]['entropy_avg'].mean()
            
            stats['Entropy_Overall'] = edf['entropy_avg'].mean()
            stats['Entropy_Pre'] = pre_ent
            stats['Entropy_Post'] = post_ent
        
        all_stats.append(stats)
        print(f"  Saved plots and data to {out_dir}")
    
    # Combined statistics across files
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(base_outputs / 'all_files_linearAR_entropy_stats.csv', index=False)
        print(f"\nSaved combined statistics: {base_outputs / 'all_files_linearAR_entropy_stats.csv'}")
    
    print("\n" + "="*70)
    print("Processing complete!")
    print(f"All outputs saved to: {base_outputs}")
    print("="*70)


if __name__ == "__main__":
    main()

