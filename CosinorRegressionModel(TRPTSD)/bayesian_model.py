import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Load your DataFrame (adjust the filename as needed)
df = pd.read_csv('data/RNS_G_M1_output.csv')  # Change to your actual file if different

# Extract time and observed value columns
t = df['X'].values
y = df['time'].values

# Set the period and frequency
period = 24
omega = 2 * np.pi / period

# Bayesian Cosinor Model
with pm.Model() as cosinor_model:
    # Priors
    M = pm.Normal('M', mu=np.mean(y), sigma=10)
    A = pm.HalfNormal('A', sigma=10)  # Amplitude should be >= 0
    phi = pm.Uniform('phi', lower=0, upper=2 * np.pi)  # Acrophase
    sigma = pm.HalfNormal('sigma', sigma=10)

    # Expected value
    mu = M + A * pm.math.cos(omega * t + phi)

    # Likelihood
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=y)

    # Sample from the posterior
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Print summary of the posterior
print(az.summary(trace, var_names=['M', 'A', 'phi', 'sigma']))

# Plot posterior distributions
az.plot_posterior(trace, var_names=['M', 'A', 'phi', 'sigma'])
plt.show()

# Optional: Plot model fit
posterior_predictive = pm.sample_posterior_predictive(trace, model=cosinor_model)
mean_pred = posterior_predictive['obs'].mean(axis=0)

plt.figure()
plt.scatter(t, y, label='Observed', color='black')
plt.plot(t, mean_pred, label='Bayesian Cosinor Fit', color='red')
plt.xlabel('Time (X)')
plt.ylabel('Observed Value (time)')
plt.legend()
plt.title('Bayesian Cosinor Regression Fit')
plt.show()

# Interpretation:
print("\nINTERPRETATION:")
print("- M (MESOR): Mean level of your rhythm.")
print("- A (Amplitude): Size of oscillation. If the 95% credible interval excludes zero, the rhythm is significant.")
print("- phi (Acrophase): Phase shift (in radians). You can convert to hours: phi * 24 / (2 * pi)")
print("- sigma: Residual standard deviation (noise level).\n")
