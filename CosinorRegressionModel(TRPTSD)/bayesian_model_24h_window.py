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

# Load pre- and post-condition data (update paths as needed)
pre_data = load_data("/Users/dyy/Documents/Project Repo/SuthanaLabComputations/CosinorRegressionModel(TRPTSD)/data/RNS_G_Pre_output.csv")
post_data = load_data("/Users/dyy/Documents/Project Repo/SuthanaLabComputations/CosinorRegressionModel(TRPTSD)/data/RNS_G_M1_output.csv")

# --- Data Preparation Functions for CosinorPy ---
def prepare_data(df):
    """Function used to prepare the data for Cosinor Regression"""
    df1 = df.copy()
    df1["test"] = df1["date"].astype(str)
    df1["x"] = df1["hour"]
    df1["y"] = df1["Pattern A Channel 2"]
    
    return df1

# --- Cosinor Model Fitting (using CosinorPy) ---
# Fit population models for pre- and post-condition data
# n_components = [1,2,3] means fitting models with 1, 2, and 3 harmonics
# period=24 for 24-hour rhythm

df_results_pre_data = cosinor.population_fit_group(prepare_data(pre_data), n_components = [1,2,3], period=24, plot=False)
df_best_models_pre_data = cosinor.get_best_models_population(prepare_data(pre_data), df_results_pre_data, n_components = [1,2,3])
cosinor.plot_df_models_population(prepare_data(pre_data), df_best_models_pre_data)
plt.savefig('cosinor_population_pre_24h.png', dpi=300, bbox_inches='tight')
plt.show()

df_results_post_data = cosinor.population_fit_group(prepare_data(post_data), n_components = [1,2,3], period=24, plot=False)
df_best_models_post_data = cosinor.get_best_models_population(prepare_data(pre_data), df_results_post_data, n_components = [1,2,3])
cosinor.plot_df_models_population(prepare_data(post_data), df_best_models_post_data)
plt.savefig('cosinor_population_post_24h.png', dpi=300, bbox_inches='tight')
plt.show()

# Combine best models from pre- and post-condition for group-level analysis
# (You may want to adjust this depending on your analysis goals)
test_statistics = pd.concat([df_best_models_pre_data, df_best_models_post_data], axis=0, ignore_index=True)

# --- Bayesian Group-Level Analysis ---
# Extract cosinor parameters for Bayesian analysis
params_df = test_statistics[['mean(amplitude)', 'mean(acrophase)', 'mean(mesor)']].copy()
amplitude = params_df['mean(amplitude)'].values
acrophase = params_df['mean(acrophase)'].values
mesor = params_df['mean(mesor)'].values

def bayesian_group_inference(data, param_name):
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
    print(f"\nPosterior summary for {param_name}:")
    print(az.summary(trace, var_names=['mu', 'sigma']))
    az.plot_posterior(trace, var_names=['mu', 'sigma'])
    plt.suptitle(f'Posterior of {param_name} - 24h Window')
    
    # Save the plot with descriptive filename
    param_clean = param_name.replace('(', '').replace(')', '').replace('mean', '').strip()
    filename = f"posterior_{param_clean}_24h_window.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Run Bayesian inference for each cosinor parameter
bayesian_group_inference(amplitude, 'mean(amplitude)')
bayesian_group_inference(acrophase, 'mean(acrophase)')
bayesian_group_inference(mesor, 'mean(mesor)')

print("\nINTERPRETATION:")
print("- For each parameter, 'mu' is the group-level mean and 'sigma' is the group-level standard deviation.")
print("- The posterior distributions and credible intervals quantify uncertainty about the population values.")
print("- For amplitude: If the credible interval for 'mu' excludes zero, the group shows a significant rhythm.")
print("- For acrophase: 'mu' gives the average phase, and 'sigma' shows its variability across the group.")
print("- For mesor: 'mu' is the average baseline level across the group.")