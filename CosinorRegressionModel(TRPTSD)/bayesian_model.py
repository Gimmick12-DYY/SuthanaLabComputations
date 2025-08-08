"""
Bayesian Group-Level Analysis of Cosinor Parameters: Pre vs Post Comparison

This script performs separate Bayesian inference on group-level cosinor parameters 
for pre- and post-condition data, allowing direct comparison between conditions.

Inputs:
    - Pre-condition data (RNS_G_Pre_output.csv)
    - Post-condition data (RNS_G_M1_output.csv)
    - Parameters analyzed: mean(amplitude), mean(acrophase), mean(mesor)

Outputs:
    - Separate posterior summaries and credible intervals for each condition
    - Posterior distribution plots for each parameter in each condition
    - Comparison plots between pre and post conditions
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin

np.seterr(divide='ignore')

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

# Load pre- and post-condition data
pre_data = load_data("/Users/dyy/Documents/Project Repo/SuthanaLabComputations/CosinorRegressionModel(TRPTSD)/data/RNS_G_Pre_output.csv")
post_data = load_data("/Users/dyy/Documents/Project Repo/SuthanaLabComputations/CosinorRegressionModel(TRPTSD)/data/RNS_G_M1_output.csv")

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

# --- Cosinor Model Fitting (Separate for Pre and Post) ---
print("Fitting cosinor models for PRE-condition data...")
df_results_pre = cosinor.population_fit_group(prepare_data_pre(pre_data), n_components=[1,2,3], period=24, plot=False)
df_best_models_pre = cosinor.get_best_models_population(prepare_data_pre(pre_data), df_results_pre, n_components=[1,2,3])

print("Fitting cosinor models for POST-condition data...")
df_results_post = cosinor.population_fit_group(prepare_data_post(post_data), n_components=[1,2,3], period=24, plot=False)
df_best_models_post = cosinor.get_best_models_population(prepare_data_post(post_data), df_results_post, n_components=[1,2,3])

# --- Extract Parameters for Each Condition ---
# Pre-condition parameters
pre_params = df_best_models_pre[['mean(amplitude)', 'mean(acrophase)', 'mean(mesor)']].copy()
pre_amplitude = pre_params['mean(amplitude)'].values
pre_acrophase = pre_params['mean(acrophase)'].values
pre_mesor = pre_params['mean(mesor)'].values

print(f"Pre-condition data shape: {df_best_models_pre.shape}")
print(f"Pre-condition parameters: {len(pre_amplitude)} samples")
print(f"Pre amplitude range: {pre_amplitude.min():.3f} to {pre_amplitude.max():.3f}")

# Post-condition parameters
post_params = df_best_models_post[['mean(amplitude)', 'mean(acrophase)', 'mean(mesor)']].copy()
post_amplitude = post_params['mean(amplitude)'].values
post_acrophase = post_params['mean(acrophase)'].values
post_mesor = post_params['mean(mesor)'].values

print(f"Post-condition data shape: {df_best_models_post.shape}")
print(f"Post-condition parameters: {len(post_amplitude)} samples")
print(f"Post amplitude range: {post_amplitude.min():.3f} to {post_amplitude.max():.3f}")

# Check if post data is empty or has issues
if len(post_amplitude) == 0:
    print("WARNING: Post-condition data is empty!")
elif np.any(np.isnan(post_amplitude)):
    print("WARNING: Post-condition data contains NaN values!")

# --- Bayesian Analysis Functions ---
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
    
    # Create and save posterior plot
    az.plot_posterior(trace, var_names=['mu', 'sigma'])
    plt.suptitle(f'Posterior of {param_name} - {condition} condition')
    
    # Save the plot with descriptive filename
    param_clean = param_name.replace('(', '').replace(')', '').replace('mean', '').strip()
    filename = f"posterior_{param_clean}_{condition.lower()}.png"
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
    ax1.set_title(f'{param_name} - Pre vs Post Comparison')
    ax1.set_ylabel(param_name)
    
    # Histogram comparison
    ax2.hist(pre_data, alpha=0.7, label='Pre', bins=10)
    ax2.hist(post_data, alpha=0.7, label='Post', bins=10)
    ax2.set_title(f'{param_name} - Distribution Comparison')
    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the comparison plot
    param_clean = param_name.replace('(', '').replace(')', '').replace('mean', '').strip()
    filename = f"comparison_{param_clean}_pre_vs_post.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# --- Run Bayesian Analysis for Each Condition ---
print("\n" + "="*50)
print("BAYESIAN ANALYSIS - PRE CONDITION")
print("="*50)
pre_amp_trace = bayesian_group_inference(pre_amplitude, 'mean(amplitude)', 'PRE')
pre_phase_trace = bayesian_group_inference(pre_acrophase, 'mean(acrophase)', 'PRE')
pre_mesor_trace = bayesian_group_inference(pre_mesor, 'mean(mesor)', 'PRE')

print("\n" + "="*50)
print("BAYESIAN ANALYSIS - POST CONDITION")
print("="*50)
post_amp_trace = bayesian_group_inference(post_amplitude, 'mean(amplitude)', 'POST')
post_phase_trace = bayesian_group_inference(post_acrophase, 'mean(acrophase)', 'POST')
post_mesor_trace = bayesian_group_inference(post_mesor, 'mean(mesor)', 'POST')

# --- Comparison Plots ---
print("\n" + "="*50)
print("COMPARISON PLOTS")
print("="*50)
compare_conditions(pre_amplitude, post_amplitude, 'mean(amplitude)')
compare_conditions(pre_acrophase, post_acrophase, 'mean(acrophase)')
compare_conditions(pre_mesor, post_mesor, 'mean(mesor)')

# --- Summary Statistics for Comparison ---
print("\n" + "="*50)
print("SUMMARY STATISTICS FOR COMPARISON")
print("="*50)

def print_comparison_stats(pre_data, post_data, param_name):
    print(f"\n{param_name}:")
    print(f"  Pre-condition:  Mean = {pre_data.mean():.3f}, Std = {pre_data.std():.3f}")
    print(f"  Post-condition: Mean = {post_data.mean():.3f}, Std = {post_data.std():.3f}")
    print(f"  Difference:     {post_data.mean() - pre_data.mean():.3f}")

print_comparison_stats(pre_amplitude, post_amplitude, 'Amplitude')
print_comparison_stats(pre_acrophase, post_acrophase, 'Acrophase')
print_comparison_stats(pre_mesor, post_mesor, 'Mesor')

print("\n" + "="*50)
print("INTERPRETATION GUIDE")
print("="*50)
print("- Compare the posterior distributions between pre and post conditions")
print("- Look for overlapping credible intervals to assess significance of differences")
print("- For amplitude: Check if both conditions show significant rhythms (CI excludes zero)")
print("- For acrophase: Compare the timing of peak activity between conditions")
print("- For mesor: Compare the baseline levels between conditions")
print("- Use the comparison plots to visualize differences in distributions") 