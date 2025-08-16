"""
Daily Cosinor Metrics Calculator and Data Mapper

This script calculates daily Cosinor metrics (one set per day) and maps them back to the original hourly data structure.
For each day, it computes mean amplitude, mean acrophase, and mean mesor, then assigns these values to all hours of that day.
The script compares different label values (e.g., B3, etc.) instead of pre/post conditions.

Inputs:
    - Single CSV file (RNS_G_Full_output.csv) with hourly data and Label column
    - CosinorPy module for rhythm analysis

Outputs:
    - Daily Cosinor metrics table for each label
    - Enhanced CSV files with daily metrics mapped to hourly structure
"""

import pandas as pd
import numpy as np
import os
from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin
from datetime import datetime, timedelta

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

def main():
    """
    Main function to process data by different label values.
    """
    # File path
    data_path = "data/RNS_G_Full_output.csv"
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/daily_metrics', exist_ok=True)
    os.makedirs('outputs/enhanced_data', exist_ok=True)
    
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
        daily_metrics.to_csv(f'outputs/daily_metrics/{label}_daily_cosinor_metrics.csv', index=False)
        enhanced_data.to_csv(f'outputs/enhanced_data/{label}_enhanced_with_cosinor_metrics.csv', index=False)
        
        # Collect for combined analysis
        all_enhanced_data.append(enhanced_data)
        all_daily_metrics.append(daily_metrics)
        
        print(f"  {len(daily_metrics)} days processed")
        print(f"  Enhanced data shape: {enhanced_data.shape}")
    
    # Combine all enhanced data
    if all_enhanced_data:
        combined_enhanced_data = pd.concat(all_enhanced_data, ignore_index=True)
        combined_enhanced_data = combined_enhanced_data.sort_values(['Label', 'Region start time'])
        combined_enhanced_data.to_csv('outputs/enhanced_data/all_labels_enhanced_with_cosinor_metrics.csv', index=False)
        
        print(f"\nCombined enhanced data shape: {combined_enhanced_data.shape}")
    
    # Combine all daily metrics
    if all_daily_metrics:
        combined_daily_metrics = pd.concat(all_daily_metrics, ignore_index=True)
        combined_daily_metrics = combined_daily_metrics.sort_values(['Label', 'date'])
        combined_daily_metrics.to_csv('outputs/daily_metrics/all_labels_daily_cosinor_metrics.csv', index=False)
        
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
    
    print(f"\nFiles saved:")
    for label in unique_labels:
        print(f"  - outputs/daily_metrics/{label}_daily_cosinor_metrics.csv")
        print(f"  - outputs/enhanced_data/{label}_enhanced_with_cosinor_metrics.csv")
    print(f"  - outputs/daily_metrics/all_labels_daily_cosinor_metrics.csv")
    print(f"  - outputs/enhanced_data/all_labels_enhanced_with_cosinor_metrics.csv")

if __name__ == "__main__":
    main() 