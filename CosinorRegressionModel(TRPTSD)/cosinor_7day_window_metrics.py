"""
7-Day Window Cosinor Metrics Calculator and Data Mapper

This script calculates Cosinor metrics over 7-day windows and maps them back to the original hourly data structure.
For each 7-day period, it computes mean amplitude, mean acrophase, and mean mesor, then assigns these values to all hours of that period.
The script compares different label values (e.g., B3, etc.) instead of pre/post conditions.

Inputs:
    - Single CSV file (RNS_G_Full_output.csv) with hourly data and Label column
    - CosinorPy module for rhythm analysis

Outputs:
    - 7-day window Cosinor metrics table for each label
    - Enhanced CSV files with 7-day metrics mapped to hourly structure
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin
from datetime import datetime, timedelta

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Create plots directory if it doesn't exist
os.makedirs('CosinorRegressionModel(TRPTSD)/plots/7day_metrics', exist_ok=True)

def load_data(path):
    """
    Load and preprocess cosinor data from a CSV file.
    Assumes the file contains a 'Region start time' column, 'Pattern A Channel 2' column, and 'Label' column.
    Returns a DataFrame with added 'date' and 'hour' columns.
    """
    df = pd.read_csv(path)
    df['Region start time'] = pd.to_datetime(df['Region start time'])
    df['date'] = df['Region start time'].dt.date
    df['hour'] = df['Region start time'].dt.hour
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

def create_7day_windows(df):
    """
    Create 7-day windows for analysis.
    Returns a list of 7-day window dataframes.
    """
    # Sort by date
    df_sorted = df.sort_values('date')
    
    # Get date range
    start_date = df_sorted['date'].min()
    end_date = df_sorted['date'].max()
    
    # Create 7-day windows
    window_data = []
    current_date = start_date
    
    while current_date <= end_date:
        window_end = current_date + timedelta(days=6)
        
        # Get data for this 7-day window
        window_mask = (df_sorted['date'] >= current_date) & (df_sorted['date'] <= window_end)
        window_df = df_sorted[window_mask].copy()
        
        if len(window_df) > 0:
            # Add window identifier
            window_df['window_start'] = current_date
            window_df['window_end'] = window_end
            window_df['window_id'] = f"{current_date}_to_{window_end}"
            
            # Prepare for cosinor analysis
            window_df['test'] = window_df['window_id']
            window_df['x'] = window_df['hour']
            window_df['y'] = window_df['Pattern A Channel 2']
            
            window_data.append(window_df)
        
        # Move to next 7-day window (non-overlapping)
        current_date = window_end + timedelta(days=1)
    
    return window_data

def calculate_7day_cosinor_metrics(window_data, period=24):
    """
    Calculate Cosinor metrics for each 7-day window.
    Returns a DataFrame with 7-day window metrics.
    """
    window_metrics = []
    
    for window_df in window_data:
        window_start = window_df['window_start'].iloc[0]
        window_end = window_df['window_end'].iloc[0]
        window_id = window_df['window_id'].iloc[0]
        
        try:
            # Fit cosinor model for this 7-day window
            results = cosinor.population_fit_group(window_df, n_components=[1,2,3], period=period, plot=False)
            best_models = cosinor.get_best_models_population(window_df, results, n_components=[1,2,3])
            
            # Extract metrics
            if not best_models.empty:
                metrics = {
                    'window_start': window_start,
                    'window_end': window_end,
                    'window_id': window_id,
                    'mean_amplitude': best_models['mean(amplitude)'].iloc[0],
                    'mean_acrophase': best_models['mean(acrophase)'].iloc[0],
                    'mean_mesor': best_models['mean(mesor)'].iloc[0],
                    'amplitude_std': best_models['std(amplitude)'].iloc[0] if 'std(amplitude)' in best_models.columns else np.nan,
                    'acrophase_std': best_models['std(acrophase)'].iloc[0] if 'std(acrophase)' in best_models.columns else np.nan,
                    'mesor_std': best_models['std(mesor)'].iloc[0] if 'std(mesor)' in best_models.columns else np.nan,
                    'p_value': best_models['p_value'].iloc[0] if 'p_value' in best_models.columns else np.nan,
                    'r_squared': best_models['r_squared'].iloc[0] if 'r_squared' in best_models.columns else np.nan,
                    'data_points': len(window_df)
                }
            else:
                metrics = {
                    'window_start': window_start,
                    'window_end': window_end,
                    'window_id': window_id,
                    'mean_amplitude': np.nan,
                    'mean_acrophase': np.nan,
                    'mean_mesor': np.nan,
                    'amplitude_std': np.nan,
                    'acrophase_std': np.nan,
                    'mesor_std': np.nan,
                    'p_value': np.nan,
                    'r_squared': np.nan,
                    'data_points': len(window_df)
                }
                
        except Exception as e:
            print(f"Error processing window {window_id}: {e}")
            metrics = {
                'window_start': window_start,
                'window_end': window_end,
                'window_id': window_id,
                'mean_amplitude': np.nan,
                'mean_acrophase': np.nan,
                'mean_mesor': np.nan,
                'amplitude_std': np.nan,
                'acrophase_std': np.nan,
                'mesor_std': np.nan,
                'p_value': np.nan,
                'r_squared': np.nan,
                'data_points': len(window_df)
            }
        
        window_metrics.append(metrics)
    
    return pd.DataFrame(window_metrics)

def map_7day_metrics_to_hourly_data(original_df, window_metrics_df):
    """
    Map 7-day window Cosinor metrics back to the original hourly data structure.
    Each hour within a 7-day window gets the same window metrics.
    """
    # Create a copy of the original dataframe
    enhanced_df = original_df.copy()
    
    # Add new columns for the metrics
    enhanced_df['window_start'] = np.nan
    enhanced_df['window_end'] = np.nan
    enhanced_df['window_id'] = np.nan
    enhanced_df['mean_amplitude_7d'] = np.nan
    enhanced_df['mean_acrophase_7d'] = np.nan
    enhanced_df['mean_mesor_7d'] = np.nan
    enhanced_df['amplitude_std_7d'] = np.nan
    enhanced_df['acrophase_std_7d'] = np.nan
    enhanced_df['mesor_std_7d'] = np.nan
    enhanced_df['p_value_7d'] = np.nan
    enhanced_df['r_squared_7d'] = np.nan
    enhanced_df['data_points_7d'] = np.nan
    
    # Map window metrics to each hour
    for _, window_row in window_metrics_df.iterrows():
        window_start = window_row['window_start']
        window_end = window_row['window_end']
        mask = (enhanced_df['date'] >= window_start) & (enhanced_df['date'] <= window_end)
        
        enhanced_df.loc[mask, 'window_start'] = window_start
        enhanced_df.loc[mask, 'window_end'] = window_end
        enhanced_df.loc[mask, 'window_id'] = window_row['window_id']
        enhanced_df.loc[mask, 'mean_amplitude_7d'] = window_row['mean_amplitude']
        enhanced_df.loc[mask, 'mean_acrophase_7d'] = window_row['mean_acrophase']
        enhanced_df.loc[mask, 'mean_mesor_7d'] = window_row['mean_mesor']
        enhanced_df.loc[mask, 'amplitude_std_7d'] = window_row['amplitude_std']
        enhanced_df.loc[mask, 'acrophase_std_7d'] = window_row['acrophase_std']
        enhanced_df.loc[mask, 'mesor_std_7d'] = window_row['mesor_std']
        enhanced_df.loc[mask, 'p_value_7d'] = window_row['p_value']
        enhanced_df.loc[mask, 'r_squared_7d'] = window_row['r_squared']
        enhanced_df.loc[mask, 'data_points_7d'] = window_row['data_points']
    
    return enhanced_df

def create_7day_metrics_plots(combined_window_metrics, combined_enhanced_data, unique_labels):
    """
    Create various plots for 7-day window Cosinor metrics analysis.
    """
    print("\nCreating plots for 7-day window metrics analysis...")
    
    # 1. Time series plots for each metric by label
    metrics_to_plot = ['mean_amplitude', 'mean_acrophase', 'mean_mesor']
    metric_names = ['Amplitude', 'Acrophase', 'Mesor']
    
    for metric, metric_name in zip(metrics_to_plot, metric_names):
        plt.figure(figsize=(15, 8))
        
        for label in unique_labels:
            label_data = combined_window_metrics[combined_window_metrics['Label'] == label]
            if len(label_data) > 0:
                # Use window_start as x-axis
                plt.plot(label_data['window_start'], label_data[metric], 
                        marker='o', linewidth=2, markersize=4, label=f'Label {label}')
        
        plt.title(f'7-Day Window {metric_name} Over Time by Label', fontsize=16, fontweight='bold')
        plt.xlabel('Window Start Date', fontsize=12)
        plt.ylabel(f'{metric_name}', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'CosinorRegressionModel(TRPTSD)/plots/7day_metrics/7day_{metric}_time_series.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Box plots comparing metrics across labels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        # Prepare data for box plot
        plot_data = []
        plot_labels = []
        
        for label in unique_labels:
            label_data = combined_window_metrics[combined_window_metrics['Label'] == label]
            if len(label_data) > 0 and label_data[metric].notna().sum() > 0:
                plot_data.append(label_data[metric].dropna())
                plot_labels.append(f'Label {label}')
        
        if plot_data:
            axes[i].boxplot(plot_data, labels=plot_labels)
            axes[i].set_title(f'7-Day {metric_name} Distribution by Label', fontweight='bold')
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('CosinorRegressionModel(TRPTSD)/plots/7day_metrics/7day_metrics_boxplots.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Heatmap of correlation between metrics
    for label in unique_labels:
        label_data = combined_window_metrics[combined_window_metrics['Label'] == label]
        if len(label_data) > 0:
            # Select numeric columns for correlation
            numeric_cols = ['mean_amplitude', 'mean_acrophase', 'mean_mesor', 'p_value', 'r_squared', 'data_points']
            available_cols = [col for col in numeric_cols if col in label_data.columns]
            
            if len(available_cols) > 1:
                corr_matrix = label_data[available_cols].corr()
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f')
                plt.title(f'7-Day Window Correlation Matrix - Label {label}', fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'CosinorRegressionModel(TRPTSD)/plots/7day_metrics/7day_correlation_matrix_label_{label}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
    
    # 4. Distribution plots for each metric
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    for i, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
        for label in unique_labels:
            label_data = combined_window_metrics[combined_window_metrics['Label'] == label]
            if len(label_data) > 0 and label_data[metric].notna().sum() > 0:
                axes[i].hist(label_data[metric].dropna(), alpha=0.7, label=f'Label {label}', bins=15)
        
        axes[i].set_title(f'7-Day Window {metric_name} Distribution by Label', fontweight='bold')
        axes[i].set_xlabel(metric_name)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('CosinorRegressionModel(TRPTSD)/plots/7day_metrics/7day_metrics_distributions.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Summary statistics plot
    plt.figure(figsize=(12, 8))
    
    # Calculate summary statistics for each label
    summary_stats = []
    for label in unique_labels:
        label_data = combined_window_metrics[combined_window_metrics['Label'] == label]
        if len(label_data) > 0:
            stats = {
                'Label': label,
                'Amplitude_Mean': label_data['mean_amplitude'].mean(),
                'Amplitude_Std': label_data['mean_amplitude'].std(),
                'Acrophase_Mean': label_data['mean_acrophase'].mean(),
                'Acrophase_Std': label_data['mean_acrophase'].std(),
                'Mesor_Mean': label_data['mean_mesor'].mean(),
                'Mesor_Std': label_data['mean_mesor'].std(),
                'Windows_Count': len(label_data)
            }
            summary_stats.append(stats)
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        
        # Create bar plot with error bars
        x = np.arange(len(summary_df))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
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
        ax.set_title('7-Day Window Summary Statistics by Label', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Label {label}' for label in summary_df['Label']])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('CosinorRegressionModel(TRPTSD)/plots/7day_metrics/7day_metrics_summary_statistics.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # 6. Window duration analysis
    plt.figure(figsize=(12, 6))
    
    for label in unique_labels:
        label_data = combined_window_metrics[combined_window_metrics['Label'] == label]
        if len(label_data) > 0:
            plt.scatter(label_data['window_start'], label_data['data_points'], 
                       alpha=0.7, label=f'Label {label}', s=50)
    
    plt.title('Data Points per 7-Day Window by Label', fontweight='bold')
    plt.xlabel('Window Start Date')
    plt.ylabel('Number of Data Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('CosinorRegressionModel(TRPTSD)/plots/7day_metrics/7day_data_points_per_window.png', 
               dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to process data by different label values with 7-day windows.
    """
    # File path
    data_path = "data/RNS_G_Full_output.csv"
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/7day_metrics', exist_ok=True)
    os.makedirs('outputs/enhanced_data_7d', exist_ok=True)
    
    # Load the full dataset
    print("Loading full dataset...")
    full_data = load_data(data_path)
    
    # Get unique labels
    unique_labels = full_data['Label'].unique()
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Process each label separately
    all_enhanced_data = []
    all_window_metrics = []
    
    for label in unique_labels:
        print(f"\nProcessing label: {label}")
        
        # Filter data for this label
        label_data = full_data[full_data['Label'] == label].copy()
        
        if len(label_data) == 0:
            print(f"No data found for label {label}")
            continue
            
        print(f"  Data shape: {label_data.shape}")
        print(f"  Date range: {label_data['date'].min()} to {label_data['date'].max()}")
        
        # Calculate 7-day window metrics for this label
        window_data = create_7day_windows(label_data)
        window_metrics = calculate_7day_cosinor_metrics(window_data, period=24)
        
        # Add label information to metrics
        window_metrics['Label'] = label
        
        # Map metrics back to hourly data
        enhanced_data = map_7day_metrics_to_hourly_data(label_data, window_metrics)
        
        # Save results for this label
        window_metrics.to_csv(f'outputs/7day_metrics/{label}_7day_cosinor_metrics.csv', index=False)
        enhanced_data.to_csv(f'outputs/enhanced_data_7d/{label}_enhanced_with_7day_cosinor_metrics.csv', index=False)
        
        # Collect for combined analysis
        all_enhanced_data.append(enhanced_data)
        all_window_metrics.append(window_metrics)
        
        print(f"  {len(window_metrics)} 7-day windows processed")
        print(f"  Enhanced data shape: {enhanced_data.shape}")
    
    # Combine all enhanced data
    if all_enhanced_data:
        combined_enhanced_data = pd.concat(all_enhanced_data, ignore_index=True)
        combined_enhanced_data = combined_enhanced_data.sort_values(['Label', 'Region start time'])
        combined_enhanced_data.to_csv('outputs/enhanced_data_7d/all_labels_enhanced_with_7day_cosinor_metrics.csv', index=False)
        
        print(f"\nCombined enhanced data shape: {combined_enhanced_data.shape}")
    
    # Combine all window metrics
    if all_window_metrics:
        combined_window_metrics = pd.concat(all_window_metrics, ignore_index=True)
        combined_window_metrics = combined_window_metrics.sort_values(['Label', 'window_start'])
        combined_window_metrics.to_csv('outputs/7day_metrics/all_labels_7day_cosinor_metrics.csv', index=False)
        
        print(f"Combined 7-day window metrics shape: {combined_window_metrics.shape}")
    
    # Create summary report
    print("\n" + "="*50)
    print("7-DAY WINDOW SUMMARY REPORT")
    print("="*50)
    
    for label in unique_labels:
        label_metrics = combined_window_metrics[combined_window_metrics['Label'] == label]
        if len(label_metrics) > 0:
            print(f"\nLabel '{label}' 7-day window metrics summary:")
            print(f"  Total 7-day windows: {len(label_metrics)}")
            print(f"  Windows with valid amplitude: {label_metrics['mean_amplitude'].notna().sum()}")
            print(f"  Windows with valid acrophase: {label_metrics['mean_acrophase'].notna().sum()}")
            print(f"  Windows with valid mesor: {label_metrics['mean_mesor'].notna().sum()}")
            
            # Show some statistics
            if label_metrics['mean_amplitude'].notna().sum() > 0:
                print(f"  Mean amplitude: {label_metrics['mean_amplitude'].mean():.3f} ± {label_metrics['mean_amplitude'].std():.3f}")
                print(f"  Mean acrophase: {label_metrics['mean_acrophase'].mean():.3f} ± {label_metrics['mean_acrophase'].std():.3f}")
                print(f"  Mean mesor: {label_metrics['mean_mesor'].mean():.3f} ± {label_metrics['mean_mesor'].std():.3f}")
    
    # Show sample of combined enhanced data
    if all_enhanced_data:
        print(f"\nSample of combined enhanced data (7-day windows):")
        sample_cols = ['Region start time', 'Pattern A Channel 2', 'Label', 'CAPS_score', 'window_id', 'mean_amplitude_7d', 'mean_acrophase_7d', 'mean_mesor_7d']
        print(combined_enhanced_data[sample_cols].head(10))
    
    # Create plots
    if all_window_metrics and all_enhanced_data:
        create_7day_metrics_plots(combined_window_metrics, combined_enhanced_data, unique_labels)
    
    print(f"\nFiles saved:")
    for label in unique_labels:
        print(f"  - outputs/7day_metrics/{label}_7day_cosinor_metrics.csv")
        print(f"  - outputs/enhanced_data_7d/{label}_enhanced_with_7day_cosinor_metrics.csv")
    print(f"  - outputs/7day_metrics/all_labels_7day_cosinor_metrics.csv")
    print(f"  - outputs/enhanced_data_7d/all_labels_enhanced_with_7day_cosinor_metrics.csv")

if __name__ == "__main__":
    main() 