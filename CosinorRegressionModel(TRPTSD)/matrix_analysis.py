import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CosinorPy import file_parser, cosinor

def load_and_reshape(path, event_col, min_valid_hours=12):
    """Function used to parser the data using the file_parser method from the CosinorPy package"""
    df = pd.read_csv(path)
    print("Available columns:", df.columns)
    df['Region start time'] = pd.to_datetime(df['Region start time'])
    df['date'] = df['Region start time'].dt.date
    df['hour'] = df['Region start time'].dt.hour
    # Pivot to days x hours
    matrix = df.pivot(index='date', columns='hour', values=event_col)
    matrix = matrix.interpolate(axis=1, limit_direction='both')
    matrix = matrix[matrix.count(axis=1) >= min_valid_hours]
    matrix = matrix.fillna(0)
    return matrix

def daily_cosinor(matrix):
    """Fit cosinor model to each row of the matrix and return R² values."""
    results = []
    for i, (date, row) in enumerate(matrix.iterrows()):
        time = np.arange(len(row))
        values = row.values
        cosinor.periodogram_df(date)
        r2 = fit['r2']
        results.append({'day': i, 'date': date, 'R2': r2})
    return pd.DataFrame(results)

def plot_metric(pre, post, metric, ylabel):
    """Plot the specified metric for pre and post conditions."""
    plt.figure(figsize=(12, 4))
    plt.plot(pre[metric], label='Pre')
    plt.plot(range(len(pre), len(pre)+len(post)), post[metric], label='Post')
    plt.axvline(len(pre), color='magenta', linestyle='--', label='Stimulus')
    plt.xlabel('Day')
    plt.ylabel(ylabel)
    plt.title(f'Daily {ylabel}')
    plt.legend()
    plt.show()

def boxplot_metric(pre, post, metric, ylabel):
    """Create a boxplot for the specified metric for pre and post conditions."""
    plt.figure(figsize=(6, 4))
    plt.boxplot([pre[metric], post[metric]], labels=['Pre', 'Post'])
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} Distribution')
    plt.show()

def print_summary(pre, post, metric, label):
    """Print summary statistics for the specified metric."""
    print(f'{label} Summary:')
    print(f'  Pre:  mean={pre[metric].mean():.2f}, std={pre[metric].std():.2f}, min={pre[metric].min():.2f}, max={pre[metric].max():.2f}')
    print(f'  Post: mean={post[metric].mean():.2f}, std={post[metric].std():.2f}, min={post[metric].min():.2f}, max={post[metric].max():.2f}')

def plot_event_heatmap(matrix, title):
    """Plot a heatmap of event detections."""
    plt.figure(figsize=(14, 6))
    ax = plt.imshow(matrix.values.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(ax, label='Event Count')
    plt.xlabel('Days')
    plt.ylabel('Time of Day (hour)')
    plt.title(title)
    plt.show()

# ---- MAIN ANALYSIS ----
PRE_PATH = 'data/RNS_G_Pre_output.csv'
POST_PATH = 'data/RNS_G_M1_output.csv'
EVENT_COL = 'Pattern A Channel 2'
MIN_VALID_HOURS = 12

pre_matrix = load_and_reshape(PRE_PATH, EVENT_COL, MIN_VALID_HOURS)
post_matrix = load_and_reshape(POST_PATH, EVENT_COL, MIN_VALID_HOURS)

pre_results = daily_cosinor(pre_matrix)
post_results = daily_cosinor(post_matrix)

# Only analyze and plot R2
for metric, label in zip(['R2'], ['R²']):
    plot_metric(pre_results, post_results, metric, label)
    boxplot_metric(pre_results, post_results, metric, label)
    print_summary(pre_results, post_results, metric, label)
    print('-'*40)

plot_event_heatmap(pd.concat([pre_matrix, post_matrix]), 'Event Detections Heatmap (Pre + Post)')