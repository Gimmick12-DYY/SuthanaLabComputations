# Bayesian Model for Cosinor Parameters
# This script performs Bayesian inference on the cosinor parameters (amplitude, acrophase, and mesor)
# It uses PyMC3 to fit a normal distribution to the parameters and provides posterior summaries and credible intervals
# The script also includes a function to run Bayesian inference for each parameter and plot the results
# The script assumes that the data is stored in a CSV file called 'cosinor_parameters.csv'
# The script also assumes that the data is normally distributed
# The script also assumes that the data is independent and identically distributed
# The script also assumes that the data is not autocorrelated
# The script also assumes that the data is not heteroscedastic
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

np.seterr(divide='ignore')
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as multi
from scipy.optimize import curve_fit
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import percentileofscore
from scipy.stats import circstd, circmean
import copy
import itertools
from random import sample
import os
import copy
from CosinorPy.helpers import df_add_row

def load_data(path):
    """Function used to parser the data using the file_parser method from the CosinorPy package"""
    df = pd.read_csv(path)
    df['Region start time'] = pd.to_datetime(df['Region start time'])
    df['date'] = df['Region start time'].dt.date
    df['hour'] = df['Region start time'].dt.hour
    df = df.drop('Unnamed: 0', axis=1)
    return df

pre_data = load_data("data/RNS_G_Pre_output.csv")
post_data = load_data("data/RNS_G_M1_output.csv")

def prepare_data_pre(df):
    """Function used to prepare the data for Cosinor Regression"""
    df1 = df.copy()
    
    # Ensure date column is in datetime format
    df1['date'] = pd.to_datetime(df1['date'])

    # Compute 'week' column by calculating the number of days since the start date
    start_date = df1['date'].min()
    df1['week'] = ((df1['date'] - start_date).dt.days // 7) + 1
    df1['week'] = df1['week'].apply(lambda x: f"week{x}")

    # Add remaining columns
    df1['test'] = df1['week'].astype(str)
    df1['x'] = df1['hour']
    df1['y'] = df1["Pattern A Channel 2"]

    return df1

def prepare_data_post(df):
    """Function used to prepare the data for Cosinor Regression"""
    df1 = df.copy()
    
    # Ensure date column is in datetime format
    df1['date'] = pd.to_datetime(df1['date'])

    # Compute 'week' column by calculating the number of days since the start date
    start_date = df1['date'].min()
    df1['week'] = ((df1['date'] - start_date).dt.days // 7) + 6
    df1['week'] = df1['week'].apply(lambda x: f"week{x}")

    # Add remaining columns
    df1['test'] = df1['week'].astype(str)
    df1['x'] = df1['hour']
    df1['y'] = df1["Pattern A Channel 2"]

    return df1

df_results_pre_data = cosinor.population_fit_group(prepare_data_pre(pre_data), n_components = [1,2,3], period=24, plot=False)
df_best_models_pre_data = cosinor.get_best_models_population(prepare_data_pre(pre_data), df_results_pre_data, n_components = [1,2,3])
cosinor.plot_df_models_population(prepare_data_pre(pre_data), df_best_models_pre_data)
df_best_models_pre_data

df_results_post_data = cosinor.population_fit_group(prepare_data_post(post_data), n_components = [1,2,3], period=24, plot=False)
df_best_models_post_data = cosinor.get_best_models_population(prepare_data_post(pre_data), df_results_post_data, n_components = [1,2,3])
cosinor.plot_df_models_population(prepare_data_post(post_data), df_best_models_post_data)
df_best_models_post_data

df_best_models_pre_data
df_best_models_post_data
test_statistics = pd.concat([df_best_models_pre_data, df_best_models_post_data], axis=0, ignore_index=True)
test_statistics


# Load the cosinor parameters DataFrame
params_df = test_statistics[['mean(amplitude)', 'mean(acrophase)', 'mean(mesor)']].copy()

# Extract the relevant columns
amplitude = params_df['mean(amplitude)'].values
acrophase = params_df['mean(acrophase)'].values
mesor = params_df['mean(mesor)'].values

# Function to run Bayesian inference for a parameter
def bayesian_group_inference(data, param_name):
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=data.mean(), sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=10)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    print(f"\nPosterior summary for {param_name}:")
    print(az.summary(trace, var_names=['mu', 'sigma']))
    az.plot_posterior(trace, var_names=['mu', 'sigma'])
    plt.suptitle(f'Posterior of {param_name}')
    plt.show()

# Run Bayesian inference for each parameter
bayesian_group_inference(amplitude, 'mean(amplitude)')
bayesian_group_inference(acrophase, 'mean(acrophase)')
bayesian_group_inference(mesor, 'mean(mesor)')

print("\nINTERPRETATION:")
print("- For each parameter, 'mu' is the group-level mean and 'sigma' is the group-level standard deviation.")
print("- The posterior distributions and credible intervals quantify uncertainty about the population values.")
print("- For amplitude: If the credible interval for 'mu' excludes zero, the group shows a significant rhythm.")
print("- For acrophase: 'mu' gives the average phase, and 'sigma' shows its variability across the group.")
print("- For mesor: 'mu' is the average baseline level across the group.")
