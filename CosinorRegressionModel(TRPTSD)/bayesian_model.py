import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Load the cosinor parameters DataFrame
params_df = pd.read_csv('cosinor_parameters.csv')

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
