"""
7-Day Window Cosinor Metrics Calculator and Data Mapper

This script calculates Cosinor metrics over 7-day windows and maps them back to the original hourly data structure.
For each 7-day period, it computes mean amplitude, mean acrophase, and mean mesor, then assigns these values to all hours of that period.

Inputs:
    - Original CSV files with hourly data
    - CosinorPy module for rhythm analysis

Outputs:
    - 7-day window Cosinor metrics table
    - Enhanced CSV files with 7-day metrics mapped to hourly structure
"""

import pandas as pd
import numpy as np
import os
from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin
from datetime import datetime, timedelta

def load_data(path):
    """
    Load and preprocess cosinor data from a CSV file.
    Assumes the file contains a 'Region start time' column and a 'Pattern A Channel 2' column.
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

def main():
    """
    Main function to process both pre and post data files with 7-day windows.
    """
    # File paths
    pre_data_path = "data/RNS_G_Pre_output.csv"
    post_data_path = "data/RNS_G_M1_output.csv"
    
    # Create output directories
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/7day_metrics', exist_ok=True)
    os.makedirs('outputs/enhanced_data_7d', exist_ok=True)
    
    # Process Pre-condition data
    print("Processing Pre-condition data with 7-day windows...")
    pre_data = load_data(pre_data_path)
    pre_window_data = create_7day_windows(pre_data)
    pre_window_metrics = calculate_7day_cosinor_metrics(pre_window_data, period=24)
    pre_enhanced_data = map_7day_metrics_to_hourly_data(pre_data, pre_window_metrics)
    
    # Save pre-condition results
    pre_window_metrics.to_csv('outputs/7day_metrics/pre_7day_cosinor_metrics.csv', index=False)
    pre_enhanced_data.to_csv('outputs/enhanced_data_7d/pre_enhanced_with_7day_cosinor_metrics.csv', index=False)
    
    print(f"Pre-condition: {len(pre_window_metrics)} 7-day windows processed")
    print(f"Pre-condition enhanced data shape: {pre_enhanced_data.shape}")
    
    # Process Post-condition data
    print("\nProcessing Post-condition data with 7-day windows...")
    post_data = load_data(post_data_path)
    post_window_data = create_7day_windows(post_data)
    post_window_metrics = calculate_7day_cosinor_metrics(post_window_data, period=24)
    post_enhanced_data = map_7day_metrics_to_hourly_data(post_data, post_window_metrics)
    
    # Save post-condition results
    post_window_metrics.to_csv('outputs/7day_metrics/post_7day_cosinor_metrics.csv', index=False)
    post_enhanced_data.to_csv('outputs/enhanced_data_7d/post_enhanced_with_7day_cosinor_metrics.csv', index=False)
    
    print(f"Post-condition: {len(post_window_metrics)} 7-day windows processed")
    print(f"Post-condition enhanced data shape: {post_enhanced_data.shape}")
    
    # Create summary report
    print("\n" + "="*50)
    print("7-DAY WINDOW SUMMARY REPORT")
    print("="*50)
    
    print(f"\nPre-condition 7-day window metrics summary:")
    print(f"  Total 7-day windows: {len(pre_window_metrics)}")
    print(f"  Windows with valid amplitude: {pre_window_metrics['mean_amplitude'].notna().sum()}")
    print(f"  Windows with valid acrophase: {pre_window_metrics['mean_acrophase'].notna().sum()}")
    print(f"  Windows with valid mesor: {pre_window_metrics['mean_mesor'].notna().sum()}")
    
    print(f"\nPost-condition 7-day window metrics summary:")
    print(f"  Total 7-day windows: {len(post_window_metrics)}")
    print(f"  Windows with valid amplitude: {post_window_metrics['mean_amplitude'].notna().sum()}")
    print(f"  Windows with valid acrophase: {post_window_metrics['mean_acrophase'].notna().sum()}")
    print(f"  Windows with valid mesor: {post_window_metrics['mean_mesor'].notna().sum()}")
    
    # Show sample of enhanced data
    print(f"\nSample of enhanced pre-condition data (7-day windows):")
    sample_cols = ['Region start time', 'Pattern A Channel 2', 'window_id', 'mean_amplitude_7d', 'mean_acrophase_7d', 'mean_mesor_7d']
    print(pre_enhanced_data[sample_cols].head(10))
    
    print(f"\nFiles saved:")
    print(f"  - outputs/7day_metrics/pre_7day_cosinor_metrics.csv")
    print(f"  - outputs/7day_metrics/post_7day_cosinor_metrics.csv")
    print(f"  - outputs/enhanced_data_7d/pre_enhanced_with_7day_cosinor_metrics.csv")
    print(f"  - outputs/enhanced_data_7d/post_enhanced_with_7day_cosinor_metrics.csv")

if __name__ == "__main__":
    main() 