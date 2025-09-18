"""
-------------------------------------------------------------------------------------------------------------------
Cosinor Regression Analysis with 24-Hour Window

Updated: Sept 2025

This script performs cosinor regression analysis on time-series data using a 24-hour window
to capture daily rhythms. It includes data loading, preparation, model fitting, and Bayesian group-level analysis.
-------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os
from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin # Some adjustment may be needed for CosinorPy (cosinor)
from pathlib import Path
import glob
import seaborn as sns


np.seterr(divide='ignore')

# --- Data Loading and Preparation ---
def load_data(path):
    """
    Load and preprocess all cosinor data from a directory.
    Able to handle multiple datasets in the directory.
    Nameing rules:
    - TRPTSD data: 'RNS_{Patient Letter}_{Pattern Name + Channel No.}_Full_output.csv
    - Example: 'RNS_A_B2_Full_output.csv' for Patient A, Pattern B Channel 2
    Assumes the file contains a 'Region start time' column, 'Pattern {} Channel {}' column, and 'Label' column.
    The function enumerates through all files in the directory and processes them into a list of DataFrames.
    Returns a single concatenated DataFrame with added normalized 'date' and 'hour' columns for further analysis.
    """

    folder_dir = Path(path)
    if folder_dir.exists() and folder_dir.is_dir():
        dataset_list = sorted([str(fp) for fp in folder_dir.glob("RNS_*_Full*.csv")])
    else:
        dataset_list = sorted(glob.glob(str(path))) or [str(path)]

    frames = []
    for dataset in dataset_list:
        data_name = Path(dataset).name
        # Extract identifiers if they follow the naming scheme
        patient_id = data_name[4] if len(data_name) > 4 else None
        pattern_channel = data_name[6:-16] if len(data_name) > 16 else None

        df = pd.read_csv(dataset)
        df['Region start time'] = pd.to_datetime(df['Region start time'])
        df['date'] = df['Region start time'].dt.date
        df['hour'] = df['Region start time'].dt.hour
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        if patient_id is not None:
            df['Patient'] = patient_id
        if pattern_channel is not None:
            df['PatternChannel'] = pattern_channel
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No datasets found for path: {path}")

    return pd.concat(frames, ignore_index=True)

def prepare_data_for_cosinor(df, y_column):
    """
    Function used to prepare the data for Cosinor Regression
    This re-label the Pattern Channel column to y and hours to x to accommodate CosinorPy function input requirements
    """
    daily_data = []
    if y_column not in df.columns:
        raise KeyError(f"Expected y column '{y_column}' not found in dataframe")

    for date, group in df.groupby('date'):
        daily_df = group.copy()
        daily_df['test'] = date.strftime('%Y-%m-%d')
        daily_df['x'] = daily_df['hour']
        daily_df['y'] = daily_df[y_column]
        daily_data.append(daily_df)
    return daily_data

def parse_dataset_info(file_path):
    """
    Parse file name like 'RNS_{Patient}_{PatternLetter}{Channel}_Full_output.csv' to extract identifiers
    Returns: key_name, y_column_name (e.g., 'A_B2' -> ('A_B2', 'Pattern B Channel 2'))
    """
    name = Path(file_path).name
    # Example: RNS_A_B2_Full_output.csv
    parts = name.split('_')
    patient = parts[1] if len(parts) > 1 else 'Unknown'
    pattern_chan = parts[2] if len(parts) > 2 else 'A2'
    pattern_letter = pattern_chan[0]
    channel_num = ''.join(ch for ch in pattern_chan[1:] if ch.isdigit()) or '2'
    key_name = f"{patient}_{pattern_letter}{channel_num}"
    y_col = f"Pattern {pattern_letter} Channel {channel_num}"
    return key_name, y_col

def create_plots_for_file(daily_metrics_df, key_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Ensure date is sorted
    dm = daily_metrics_df.copy()
    dm = dm.sort_values('date')

    metrics = [('mean_amplitude', 'Amplitude'), ('mean_acrophase', 'Acrophase'), ('mean_mesor', 'Mesor')]
    # Determine label column for coloring (prefer 'Label', then 'label', else derive YYYY-MM)
    label_col = 'Label' if 'Label' in dm.columns else ('label' if 'label' in dm.columns else None)
    if label_col is None:
        dm['Label'] = pd.to_datetime(dm['date']).astype('datetime64[ns]').dt.strftime('%Y-%m')
        label_col = 'Label'

    unique_labels = [lab for lab in dm[label_col].dropna().unique()]
    palette = sns.color_palette('tab10', n_colors=max(3, len(unique_labels) or 3))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}

    for col, metric_name in metrics:
        plt.figure(figsize=(12, 6))
        if unique_labels:
            for lab in unique_labels:
                sub = dm[dm[label_col] == lab].sort_values('date')
                if sub.empty:
                    continue
                plt.plot(sub['date'], sub[col], marker='o', linewidth=2, label=str(lab), color=color_map[lab])
            plt.legend(title='Label', fontsize=9)
        else:
            plt.plot(dm['date'], dm[col], marker='o', linewidth=2)
        plt.title(f'{metric_name} Over Time - {key_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{key_name}_daily_{col}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Distributions
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    for ax, (col, label) in zip(axes, metrics):
        data = dm[col].dropna()
        if len(data) > 0:
            ax.hist(data, bins=15, alpha=0.8)
        ax.set_title(f'{label} Distribution - {key_name}', fontweight='bold')
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{key_name}_daily_metrics_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Correlation heatmap (if enough data)
    numeric_cols = ['mean_amplitude', 'mean_acrophase', 'mean_mesor', 'p_value', 'r_squared']
    available_cols = [c for c in numeric_cols if c in dm.columns]
    if len(available_cols) > 1 and dm[available_cols].dropna().shape[0] > 1:
        corr = dm[available_cols].corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f')
        plt.title(f'Correlation Matrix - {key_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{key_name}_correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

# --- Model calculation ---
def calculate_daily_cosinor_metrics(daily_data, period=24):
    """
    Calculate Cosinor metrics for each day.
    Returns a DataFrame with daily metrics.
    Parameters:
    daily_data (list of pd.DataFrame): List of daily dataframes.
    period (int): Period for the cosinor model, default is 24 hours.
    24-hour period is used to capture daily rhythms.
    """
    daily_metrics = []
    
    for day_data in daily_data:
        date = day_data['date'].iloc[0]
        
        try:
            # Fit cosinor model for this day
            results = cosinor.population_fit_group(day_data, n_components=[1,2,3], period=period, plot=False)
            best_models = cosinor.get_best_models_population(day_data, results, n_components=[1,2,3])
            
            # Extract metrics
            # Problem: due to CosinorPY limits, since we only have one observation per hour, p-values, R-squared, and standard deviations may not be available.
            if not best_models.empty:
                metrics = {
                    'date': date,
                    'mean_amplitude': best_models['mean(amplitude)'].iloc[0],
                    'mean_acrophase': best_models['mean(acrophase)'].iloc[0],
                    'mean_mesor': best_models['mean(mesor)'].iloc[0],
                    'amplitude_std': best_models['std(amplitude)'].iloc[0] if 'std(amplitude)' in best_models.columns else np.nan,
                    'acrophase_std': best_models['std(acrophase)'].iloc[0] if 'std(acrophase)' in best_models.columns else np.nan,
                    'mesor_std': best_models['std(mesor)'].iloc[0] if 'std(mesor)' in best_models.columns else np.nan,
                    'p_value': best_models['p_value'].iloc[0] if 'p_value' in best_models.columns else np.nan,
                    'r_squared': best_models['r_squared'].iloc[0] if 'r_squared' in best_models.columns else np.nan
                }
            else:
                metrics = {
                    'date': date,
                    'mean_amplitude': np.nan,
                    'mean_acrophase': np.nan,
                    'mean_mesor': np.nan,
                    'amplitude_std': np.nan,
                    'acrophase_std': np.nan,
                    'mesor_std': np.nan,
                    'p_value': np.nan,
                    'r_squared': np.nan
                }
                
        except Exception as e:
            print(f"Error processing date {date}: {e}")
            metrics = {
                'date': date,
                'mean_amplitude': np.nan,
                'mean_acrophase': np.nan,
                'mean_mesor': np.nan,
                'amplitude_std': np.nan,
                'acrophase_std': np.nan,
                'mesor_std': np.nan,
                'p_value': np.nan,
                'r_squared': np.nan
            }
        
        daily_metrics.append(metrics)
    
    return pd.DataFrame(daily_metrics)

def map_metrics_to_hourly_data(original_df, daily_metrics_df):
    """
    Map daily Cosinor metrics back to the original hourly data structure.
    Each hour of a day gets the same daily metrics.
    """
    # Create a copy of the original dataframe
    enhanced_df = original_df.copy()
    
    # Add new columns for the metrics
    enhanced_df['mean_amplitude'] = np.nan
    enhanced_df['mean_acrophase'] = np.nan
    enhanced_df['mean_mesor'] = np.nan
    enhanced_df['amplitude_std'] = np.nan
    enhanced_df['acrophase_std'] = np.nan
    enhanced_df['mesor_std'] = np.nan
    enhanced_df['p_value'] = np.nan
    enhanced_df['r_squared'] = np.nan
    
    # Map daily metrics to each hour
    for _, daily_row in daily_metrics_df.iterrows():
        date = daily_row['date']
        mask = enhanced_df['date'] == date
        
        enhanced_df.loc[mask, 'mean_amplitude'] = daily_row['mean_amplitude']
        enhanced_df.loc[mask, 'mean_acrophase'] = daily_row['mean_acrophase']
        enhanced_df.loc[mask, 'mean_mesor'] = daily_row['mean_mesor']
        enhanced_df.loc[mask, 'amplitude_std'] = daily_row['amplitude_std']
        enhanced_df.loc[mask, 'acrophase_std'] = daily_row['acrophase_std']
        enhanced_df.loc[mask, 'mesor_std'] = daily_row['mesor_std']
        enhanced_df.loc[mask, 'p_value'] = daily_row['p_value']
        enhanced_df.loc[mask, 'r_squared'] = daily_row['r_squared']
    
    return enhanced_df

# --- Model visualizations ---
def create_daily_metrics_plots(combined_daily_metrics, combined_enhanced_data, unique_labels):
    """
    Create various plots for daily Cosinor metrics analysis.
    """
    print("\nCreating plots for daily metrics analysis...")
    
    # 1. Time series plots for each metric by label
    metrics_to_plot = ['mean_amplitude', 'mean_acrophase', 'mean_mesor']
    metric_names = ['Amplitude', 'Acrophase', 'Mesor']
    
    # Dynamically scale figure width by date span to make x-axis longer when needed
    try:
        num_days = pd.to_datetime(combined_daily_metrics['date']).nunique()
    except Exception:
        num_days = combined_daily_metrics['date'].nunique()
    base_width = max(24, int(num_days / 3))
    fig_width = max(2400, base_width * 2)  # 2x wider, with a higher cap

    for metric, metric_name in zip(metrics_to_plot, metric_names):
        plt.figure(figsize=(200, 8))
        
        for label in unique_labels:
            label_data = combined_daily_metrics[combined_daily_metrics['Label'] == label]
            if len(label_data) > 0:
                # Downsample to avoid overplotting if too many points
                n = len(label_data)
                step = max(1, n // 3000)
                plt.plot(label_data['date'].iloc[::step], label_data[metric].iloc[::step],
                        marker='o', linewidth=1.2, markersize=2.5, label=f'Label {label}', alpha=0.9)
        
        plt.title(f'Daily {metric_name} Over Time by Label', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'{metric_name}', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'CosinorRegressionModel(TRPTSD)/plots/daily_metrics/daily_{metric}_time_series.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Box plots comparing metrics across labels
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    for i, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        # Prepare data for box plot
        plot_data = []
        plot_labels = []
        
        for label in unique_labels:
            label_data = combined_daily_metrics[combined_daily_metrics['Label'] == label]
            if len(label_data) > 0 and label_data[metric].notna().sum() > 0:
                plot_data.append(label_data[metric].dropna())
                plot_labels.append(f'Label {label}')
        
        if plot_data:
            axes[i].boxplot(plot_data, labels=plot_labels)
            axes[i].set_title(f'{metric_name} Distribution by Label', fontweight='bold')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('CosinorRegressionModel(TRPTSD)/plots/daily_metrics/daily_metrics_boxplots.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Heatmap of correlation between metrics
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix for each label
    for label in unique_labels:
        label_data = combined_daily_metrics[combined_daily_metrics['Label'] == label]
        if len(label_data) > 0:
            # Select numeric columns for correlation
            numeric_cols = ['mean_amplitude', 'mean_acrophase', 'mean_mesor', 'p_value', 'r_squared']
            available_cols = [col for col in numeric_cols if col in label_data.columns]
            
            if len(available_cols) > 1:
                corr_matrix = label_data[available_cols].corr()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f')
                plt.title(f'Correlation Matrix - Label {label}', fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'CosinorRegressionModel(TRPTSD)/plots/daily_metrics/correlation_matrix_label_{label}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
    
    # 4. Distribution plots for each metric
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    
    for i, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        for label in unique_labels:
            label_data = combined_daily_metrics[combined_daily_metrics['Label'] == label]
            if len(label_data) > 0 and label_data[metric].notna().sum() > 0:
                axes[i].hist(label_data[metric].dropna(), alpha=0.7, label=f'Label {label}', bins=15)
        
        axes[i].set_title(f'{metric_name} Distribution by Label', fontweight='bold')
        axes[i].set_xlabel(metric_name)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('CosinorRegressionModel(TRPTSD)/plots/daily_metrics/daily_metrics_distributions.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Summary statistics plot
    plt.figure(figsize=(24, 8))
    
    # Calculate summary statistics for each label
    summary_stats = []
    for label in unique_labels:
        label_data = combined_daily_metrics[combined_daily_metrics['Label'] == label]
        if len(label_data) > 0:
            stats = {
                'Label': label,
                'Amplitude_Mean': label_data['mean_amplitude'].mean(),
                'Amplitude_Std': label_data['mean_amplitude'].std(),
                'Acrophase_Mean': label_data['mean_acrophase'].mean(),
                'Acrophase_Std': label_data['mean_acrophase'].std(),
                'Mesor_Mean': label_data['mean_mesor'].mean(),
                'Mesor_Std': label_data['mean_mesor'].std(),
                'Days_Count': len(label_data)
            }
            summary_stats.append(stats)
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        
        # Create bar plot with error bars
        x = np.arange(len(summary_df))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(24, 8))
        
        # Amplitude
        ax.bar(x - width, summary_df['Amplitude_Mean'], width, 
               label='Amplitude', yerr=summary_df['Amplitude_Std'], capsize=5)
        
        # Acrophase
        ax.bar(x, summary_df['Acrophase_Mean'], width, 
               label='Acrophase', yerr=summary_df['Acrophase_Std'], capsize=5)
        
        # Mesor
        ax.bar(x + width, summary_df['Mesor_Mean'], width, 
               label='Mesor', yerr=summary_df['Mesor_Std'], capsize=5)
        
        ax.set_xlabel('Label')
        ax.set_ylabel('Metric Value')
        ax.set_title('Summary Statistics by Label', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Label {label}' for label in summary_df['Label']])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('CosinorRegressionModel(TRPTSD)/plots/daily_metrics/daily_metrics_summary_statistics.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to process data by different label values.
    """
    # Data path: directory with multiple CSVs or a single CSV path
    data_path = "CosinorRegressionModel(TRPTSD)/data/Data_Cosinor_09.15.25"
    
    # Create output directories
    os.makedirs('CosinorRegressionModel(TRPTSD)/outputs', exist_ok=True)
    os.makedirs('CosinorRegressionModel(TRPTSD)/outputs/daily_metrics', exist_ok=True)
    os.makedirs('CosinorRegressionModel(TRPTSD)/outputs/enhanced_data', exist_ok=True)
    os.makedirs('CosinorRegressionModel(TRPTSD)/plots/daily_metrics', exist_ok=True)
    
    # Enumerate CSVs and process each file individually
    folder_dir = Path(data_path)
    dataset_list = sorted([str(fp) for fp in folder_dir.glob("RNS_*_Full*.csv")])
    if not dataset_list:
        # Fallback to any CSVs
        dataset_list = sorted([str(fp) for fp in folder_dir.glob("*.csv")])
    if not dataset_list:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    print(f"Found {len(dataset_list)} datasets to process.")

    for dataset in dataset_list:
        key_name, y_col = parse_dataset_info(dataset)
        print(f"\nProcessing dataset: {Path(dataset).name} -> key '{key_name}', y column '{y_col}'")

        df = pd.read_csv(dataset)
        df['Region start time'] = pd.to_datetime(df['Region start time'])
        df['date'] = df['Region start time'].dt.date
        df['hour'] = df['Region start time'].dt.hour
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        # Prepare and compute metrics
        daily_data = prepare_data_for_cosinor(df, y_col)
        daily_metrics = calculate_daily_cosinor_metrics(daily_data, period=24)
        # Propagate label (month) into metrics for colored plots
        if 'Label' in df.columns:
            # Map each date's majority label
            label_by_date = df.groupby('date')['Label'].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            daily_metrics['Label'] = daily_metrics['date'].map(label_by_date)
        elif 'label' in df.columns:
            label_by_date = df.groupby('date')['label'].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            daily_metrics['Label'] = daily_metrics['date'].map(label_by_date)
        else:
            # Derive YYYY-MM from date
            daily_metrics['Label'] = pd.to_datetime(daily_metrics['date']).astype('datetime64[ns]').dt.strftime('%Y-%m')
        enhanced_data = map_metrics_to_hourly_data(df, daily_metrics)

        # Output directories per dataset
        per_out_dir = os.path.join('CosinorRegressionModel(TRPTSD)', 'outputs', key_name)
        os.makedirs(per_out_dir, exist_ok=True)

        # Save spreadsheets
        daily_metrics.to_csv(os.path.join(per_out_dir, f'{key_name}_daily_cosinor_metrics.csv'), index=False)
        enhanced_data.to_csv(os.path.join(per_out_dir, f'{key_name}_enhanced_with_cosinor_metrics.csv'), index=False)

        # Also preserve previous structure outputs (optional)
        os.makedirs('CosinorRegressionModel(TRPTSD)/outputs/daily_metrics', exist_ok=True)
        os.makedirs('CosinorRegressionModel(TRPTSD)/outputs/enhanced_data', exist_ok=True)
        daily_metrics.to_csv(f'CosinorRegressionModel(TRPTSD)/outputs/daily_metrics/{key_name}_daily_cosinor_metrics.csv', index=False)
        enhanced_data.to_csv(f'CosinorRegressionModel(TRPTSD)/outputs/enhanced_data/{key_name}_enhanced_with_cosinor_metrics.csv', index=False)

        # Plots per dataset
        plots_dir = os.path.join(per_out_dir, 'plots')
        create_plots_for_file(daily_metrics, key_name, plots_dir)

        print(f"  Saved results to {per_out_dir}")

if __name__ == "__main__":
    main() 