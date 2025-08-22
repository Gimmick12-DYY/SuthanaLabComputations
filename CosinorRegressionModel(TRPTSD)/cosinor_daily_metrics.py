"""
Daily Cosinor Metrics Calculator and Data Mapper

This script calculates daily Cosinor metrics (one set per day) and maps them back to the original hourly data structure.
For each day, it computes mean amplitude, mean acrophase, and mean mesor, then assigns these values to all hours of that day.
The script compares different label values (e.g., B3, etc.) instead of pre/post conditions.

Inputs:
    - Single CSV file (RNS_G_Full_output.csv) with hourly RNS data and Label column
    - CosinorPy module for rhythm analysis

Outputs:
    - Daily Cosinor metrics table for each label
    - Enhanced CSV files with daily metrics mapped to hourly structure
    - Analysis plots for distribution comparison
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
os.makedirs('CosinorRegressionModel(TRPTSD)/plots/daily_metrics', exist_ok=True)

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

def prepare_data_for_cosinor(df):
    """
    Prepare data for Cosinor analysis by grouping by date.
    Returns a list of daily dataframes.
    """
    daily_data = []
    for date, group in df.groupby('date'):
        daily_df = group.copy()
        daily_df['test'] = date.strftime('%Y-%m-%d')
        daily_df['x'] = daily_df['hour']
        daily_df['y'] = daily_df['Pattern A Channel 2']
        daily_data.append(daily_df)
    return daily_data

def calculate_daily_cosinor_metrics(daily_data, period=24):
    """
    Calculate Cosinor metrics for each day.
    Returns a DataFrame with daily metrics.
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

def create_daily_metrics_plots(combined_daily_metrics, combined_enhanced_data, unique_labels):
    """
    Create various plots for daily Cosinor metrics analysis.
    """
    print("\nCreating plots for daily metrics analysis...")
    
    # 1. Time series plots for each metric by label
    metrics_to_plot = ['mean_amplitude', 'mean_acrophase', 'mean_mesor']
    metric_names = ['Amplitude', 'Acrophase', 'Mesor']
    
    for metric, metric_name in zip(metrics_to_plot, metric_names):
        plt.figure(figsize=(15, 8))
        
        for label in unique_labels:
            label_data = combined_daily_metrics[combined_daily_metrics['Label'] == label]
            if len(label_data) > 0:
                plt.plot(label_data['date'], label_data[metric], 
                        marker='o', linewidth=2, markersize=4, label=f'Label {label}')
        
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
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
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, fmt='.2f')
                plt.title(f'Correlation Matrix - Label {label}', fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'CosinorRegressionModel(TRPTSD)/plots/daily_metrics/correlation_matrix_label_{label}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
    
    # 4. Distribution plots for each metric
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
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
    plt.figure(figsize=(12, 8))
    
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
    # File path
    data_path = "CosinorRegressionModel(TRPTSD)/data/RNS_G_Full_output.csv"
    
    # Create output directories
    os.makedirs('CosinorRegressionModel(TRPTSD)/outputs', exist_ok=True)
    os.makedirs('CosinorRegressionModel(TRPTSD)/outputs/daily_metrics', exist_ok=True)
    os.makedirs('CosinorRegressionModel(TRPTSD)/outputs/enhanced_data', exist_ok=True)
    
    # Load the full dataset
    print("Loading full dataset...")
    full_data = load_data(data_path)
    
    # Get unique labels
    unique_labels = full_data['Label'].unique()
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Process each label separately
    all_enhanced_data = []
    all_daily_metrics = []
    
    for label in unique_labels:
        print(f"\nProcessing label: {label}")
        
        # Filter data for this label
        label_data = full_data[full_data['Label'] == label].copy()
        
        if len(label_data) == 0:
            print(f"No data found for label {label}")
            continue
            
        print(f"  Data shape: {label_data.shape}")
        print(f"  Date range: {label_data['date'].min()} to {label_data['date'].max()}")
        
        # Calculate daily metrics for this label
        daily_data = prepare_data_for_cosinor(label_data)
        daily_metrics = calculate_daily_cosinor_metrics(daily_data, period=24)
        
        # Add label information to metrics
        daily_metrics['Label'] = label
        
        # Map metrics back to hourly data
        enhanced_data = map_metrics_to_hourly_data(label_data, daily_metrics)
        
        # Save results for this label
        daily_metrics.to_csv(f'CosinorRegressionModel(TRPTSD)/outputs/daily_metrics/{label}_daily_cosinor_metrics.csv', index=False)
        enhanced_data.to_csv(f'CosinorRegressionModel(TRPTSD)/outputs/enhanced_data/{label}_enhanced_with_cosinor_metrics.csv', index=False)
        
        # Collect for combined analysis
        all_enhanced_data.append(enhanced_data)
        all_daily_metrics.append(daily_metrics)
        
        print(f"  {len(daily_metrics)} days processed")
        print(f"  Enhanced data shape: {enhanced_data.shape}")
    
    # Combine all enhanced data
    if all_enhanced_data:
        combined_enhanced_data = pd.concat(all_enhanced_data, ignore_index=True)
        combined_enhanced_data = combined_enhanced_data.sort_values(['Label', 'Region start time'])
        combined_enhanced_data.to_csv('CosinorRegressionModel(TRPTSD)/outputs/enhanced_data/all_labels_enhanced_with_cosinor_metrics.csv', index=False)
        
        print(f"\nCombined enhanced data shape: {combined_enhanced_data.shape}")
    
    # Combine all daily metrics
    if all_daily_metrics:
        combined_daily_metrics = pd.concat(all_daily_metrics, ignore_index=True)
        combined_daily_metrics = combined_daily_metrics.sort_values(['Label', 'date'])
        combined_daily_metrics.to_csv('CosinorRegressionModel(TRPTSD)/outputs/daily_metrics/all_labels_daily_cosinor_metrics.csv', index=False)
        
        print(f"Combined daily metrics shape: {combined_daily_metrics.shape}")
    
    # Create summary report
    print("\n" + "="*50)
    print("SUMMARY REPORT")
    print("="*50)
    
    for label in unique_labels:
        label_metrics = combined_daily_metrics[combined_daily_metrics['Label'] == label]
        if len(label_metrics) > 0:
            print(f"\nLabel '{label}' daily metrics summary:")
            print(f"  Total days: {len(label_metrics)}")
            print(f"  Days with valid amplitude: {label_metrics['mean_amplitude'].notna().sum()}")
            print(f"  Days with valid acrophase: {label_metrics['mean_acrophase'].notna().sum()}")
            print(f"  Days with valid mesor: {label_metrics['mean_mesor'].notna().sum()}")
            
            # Show some statistics
            if label_metrics['mean_amplitude'].notna().sum() > 0:
                print(f"  Mean amplitude: {label_metrics['mean_amplitude'].mean():.3f} ± {label_metrics['mean_amplitude'].std():.3f}")
                print(f"  Mean acrophase: {label_metrics['mean_acrophase'].mean():.3f} ± {label_metrics['mean_acrophase'].std():.3f}")
                print(f"  Mean mesor: {label_metrics['mean_mesor'].mean():.3f} ± {label_metrics['mean_mesor'].std():.3f}")
    
    # Show sample of combined enhanced data
    if all_enhanced_data:
        print(f"\nSample of combined enhanced data:")
        sample_cols = ['Region start time', 'Pattern A Channel 2', 'Label', 'CAPS_score', 'mean_amplitude', 'mean_acrophase', 'mean_mesor']
        print(combined_enhanced_data[sample_cols].head(10))
    
    # Create plots
    if all_daily_metrics and all_enhanced_data:
        create_daily_metrics_plots(combined_daily_metrics, combined_enhanced_data, unique_labels)
    
    print(f"\nFiles saved:")
    for label in unique_labels:
        print(f"  - CosinorRegressionModel(TRPTSD)/outputs/daily_metrics/{label}_daily_cosinor_metrics.csv")
        print(f"  - CosinorRegressionModel(TRPTSD)/outputs/enhanced_data/{label}_enhanced_with_cosinor_metrics.csv")
    print(f"  - CosinorRegressionModel(TRPTSD)/outputs/daily_metrics/all_labels_daily_cosinor_metrics.csv")
    print(f"  - CosinorRegressionModel(TRPTSD)/outputs/enhanced_data/all_labels_enhanced_with_cosinor_metrics.csv")

if __name__ == "__main__":
    main() 