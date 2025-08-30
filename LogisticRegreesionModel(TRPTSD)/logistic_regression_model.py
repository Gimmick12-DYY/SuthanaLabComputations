"""
This module contains the implementation of a logistic regression model for predicting LFP data from TRPTSD data.
This module is part of the TRPTSD project for the Suthana Lab Group.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# Data Loading and Preprocessing
def load_data(trptsd_path, lfp_path):
    """
    Load TRPTSD and LFP data from CSV files.
    
    Parameters:
    trptsd_path (str): Path to the TRPTSD data CSV file.
    lfp_path (str): Path to the LFP data CSV file.
    
    Returns:
    pd.DataFrame, pd.DataFrame: Loaded TRPTSD and LFP data.
    """
    trptsd_data = pd.read_csv(trptsd_path)
    lfp_data = pd.read_csv(lfp_path)
    return trptsd_data, lfp_data