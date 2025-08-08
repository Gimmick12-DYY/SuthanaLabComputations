"""
Bayesian Group-Level Analysis of Cosinor Parameters

This script performs Bayesian inference on group-level cosinor parameters (amplitude, acrophase, mesor)
using PyMC. It fits a normal distribution to each parameter across subjects or samples, estimates the
posterior distributions for the group mean and standard deviation, and visualizes the results.

Inputs:
    - Cosinor model results from pre- and post-condition data (using CosinorPy module)
    - Parameters analyzed: mean(amplitude), mean(acrophase), mean(mesor)

Outputs (PyMC):
    - Posterior summaries and credible intervals for group mean and standard deviation of each parameter
    - Posterior distribution plots for each parameter
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import os
from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin

np.seterr(divide='ignore')

# Create plots directory if it doesn't exist
os.makedirs('plots/7ds_window', exist_ok=True)

# --- Data Loading and Preparation ---
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
    df = df.drop('Unnamed: 0', axis=1)
    return df

# Load pre- and post-condition data (update paths as needed)
pre_data = load_data("/Users/dyy/Documents/Project Repo/SuthanaLabComputations/CosinorRegressionModel(TRPTSD)/data/RNS_G_Pre_output.csv")
post_data = load_data("/Users/dyy/Documents/Project Repo/SuthanaLabComputations/CosinorRegressionModel(TRPTSD)/data/RNS_G_M1_output.csv")

# --- Data Preparation Functions for CosinorPy ---
def prepare_data_pre(df):
    """
    Prepare pre-condition data for cosinor regression.
    Adds 'week', 'test', 'x' (hour), and 'y' (signal) columns.
    """
    df1 = df.copy()
    df1['date'] = pd.to_datetime(df1['date'])
    start_date = df1['date'].min()
    df1['week'] = ((df1['date'] - start_date).dt.days // 7) + 1
    df1['week'] = df1['week'].apply(lambda x: f"week{x}")
    df1['test'] = df1['week'].astype(str)
    df1['x'] = df1['hour']
    df1['y'] = df1["Pattern A Channel 2"]
    return df1

def prepare_data_post(df):
    """
    Prepare post-condition data for cosinor regression.
    Adds 'week', 'test', 'x' (hour), and 'y' (signal) columns.
    """
    df1 = df.copy()
    df1['date'] = pd.to_datetime(df1['date'])
    start_date = df1['date'].min()
    df1['week'] = ((df1['date'] - start_date).dt.days // 7) + 6
    df1['week'] = df1['week'].apply(lambda x: f"week{x}")
    df1['test'] = df1['week'].astype(str)
    df1['x'] = df1['hour']
    df1['y'] = df1["Pattern A Channel 2"]
    return df1

# --- Cosinor Model Fitting (using CosinorPy) ---
# Fit population models for pre- and post-condition data
# n_components = [1,2,3] means fitting models with 1, 2, and 3 harmonics
# period=24 for 24-hour rhythm

df_results_pre_data = cosinor.population_fit_group(prepare_data_pre(pre_data), n_components=[1,2,3], period=24, plot=False)
df_best_models_pre_data = cosinor.get_best_models_population(prepare_data_pre(pre_data), df_results_pre_data, n_components=[1,2,3])
cosinor.plot_df_models_population(prepare_data_pre(pre_data), df_best_models_pre_data)
plt.savefig('CosinorRegressionModel(TRPTSD)/plots/7ds_window/cosinor_population_pre_7ds.png', dpi=300, bbox_inches='tight')
plt.show()

df_results_post_data = cosinor.population_fit_group(prepare_data_post(post_data), n_components=[1,2,3], period=24, plot=False)
df_best_models_post_data = cosinor.get_best_models_population(prepare_data_post(pre_data), df_results_post_data, n_components=[1,2,3])
cosinor.plot_df_models_population(prepare_data_post(post_data), df_best_models_post_data)
plt.savefig('CosinorRegressionModel(TRPTSD)/plots/7ds_window/cosinor_population_post_7ds.png', dpi=300, bbox_inches='tight')
plt.show()

# Combine best models from pre- and post-condition for group-level analysis
# (You may want to adjust this depending on your analysis goals)
test_statistics = pd.concat([df_best_models_pre_data, df_best_models_post_data], axis=0, ignore_index=True)

# --- Bayesian Group-Level Analysis ---
# Extract cosinor parameters for Bayesian analysis - SEPARATE for pre and post

# Pre-condition parameters
pre_params = df_best_models_pre_data[['mean(amplitude)', 'mean(acrophase)', 'mean(mesor)']].copy()
pre_amplitude = pre_params['mean(amplitude)'].values
pre_acrophase = pre_params['mean(acrophase)'].values
pre_mesor = pre_params['mean(mesor)'].values

print(f"Pre-condition data shape: {df_best_models_pre_data.shape}")
print(f"Pre-condition parameters: {len(pre_amplitude)} samples")

# Post-condition parameters
post_params = df_best_models_post_data[['mean(amplitude)', 'mean(acrophase)', 'mean(mesor)']].copy()
post_amplitude = post_params['mean(amplitude)'].values
post_acrophase = post_params['mean(acrophase)'].values
post_mesor = post_params['mean(mesor)'].values

print(f"Post-condition data shape: {df_best_models_post_data.shape}")
print(f"Post-condition parameters: {len(post_amplitude)} samples")

def bayesian_group_inference(data, param_name, condition):
    """
    Perform Bayesian inference for a group-level cosinor parameter.
    Fits a normal distribution to the data and samples the posterior for mean (mu) and std (sigma).
    Plots and prints the posterior distributions and credible intervals.
    """
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=data.mean(), sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=10)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    print(f"\nPosterior summary for {param_name} - {condition} condition:")
    print(az.summary(trace, var_names=['mu', 'sigma']))
    az.plot_posterior(trace, var_names=['mu', 'sigma'])
    plt.suptitle(f'Posterior of {param_name} - {condition} condition (7 Days Window)')
    
    # Save the plot with descriptive filename
    param_clean = param_name.replace('(', '').replace(')', '').replace('mean', '').strip()
    filename = f"CosinorRegressionModel(TRPTSD)/plots/7ds_window/posterior_{param_clean}_{condition.lower()}_7ds_window.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return trace

def compare_conditions(pre_data, post_data, param_name):
    """
    Create comparison plots between pre and post conditions for a given parameter.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Box plot comparison
    ax1.boxplot([pre_data, post_data], labels=['Pre', 'Post'])
    ax1.set_title(f'{param_name} - Pre vs Post Comparison (7 Days Window)')
    ax1.set_ylabel(param_name)
    
    # Histogram comparison
    ax2.hist(pre_data, alpha=0.7, label='Pre', bins=10)
    ax2.hist(post_data, alpha=0.7, label='Post', bins=10)
    ax2.set_title(f'{param_name} - Distribution Comparison (7 Days Window)')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the comparison plot
    param_clean = param_name.replace('(', '').replace(')', '').replace('mean', '').strip()
    filename = f"CosinorRegressionModel(TRPTSD)/plots/7ds_window/comparison_{param_clean}_pre_vs_post_7ds_window.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# --- Run Bayesian Analysis for Each Condition ---
print("\n" + "="*50)
print("BAYESIAN ANALYSIS - PRE CONDITION (7 Days Window)")
print("="*50)
pre_amp_trace = bayesian_group_inference(pre_amplitude, 'mean(amplitude)', 'PRE')
pre_phase_trace = bayesian_group_inference(pre_acrophase, 'mean(acrophase)', 'PRE')
pre_mesor_trace = bayesian_group_inference(pre_mesor, 'mean(mesor)', 'PRE')

print("\n" + "="*50)
print("BAYESIAN ANALYSIS - POST CONDITION (7 Days Window)")
print("="*50)
post_amp_trace = bayesian_group_inference(post_amplitude, 'mean(amplitude)', 'POST')
post_phase_trace = bayesian_group_inference(post_acrophase, 'mean(acrophase)', 'POST')
post_mesor_trace = bayesian_group_inference(post_mesor, 'mean(mesor)', 'POST')

# --- Comparison Plots ---
print("\n" + "="*50)
print("COMPARISON PLOTS (7 Days Window)")
print("="*50)
compare_conditions(pre_amplitude, post_amplitude, 'mean(amplitude)')
compare_conditions(pre_acrophase, post_acrophase, 'mean(acrophase)')
compare_conditions(pre_mesor, post_mesor, 'mean(mesor)')

# --- Summary Statistics for Comparison ---
print("\n" + "="*50)
print("SUMMARY STATISTICS FOR COMPARISON (7 Days Window)")
print("="*50)

def print_comparison_stats(pre_data, post_data, param_name):
    print(f"\n{param_name}:")
    print(f"  Pre-condition:  Mean = {pre_data.mean():.3f}, Std = {pre_data.std():.3f}")
    print(f"  Post-condition: Mean = {post_data.mean():.3f}, Std = {post_data.std():.3f}")
    print(f"  Difference:     {post_data.mean() - pre_data.mean():.3f}")

print_comparison_stats(pre_amplitude, post_amplitude, 'Amplitude')
print_comparison_stats(pre_acrophase, post_acrophase, 'Acrophase')
print_comparison_stats(pre_mesor, post_mesor, 'Mesor')