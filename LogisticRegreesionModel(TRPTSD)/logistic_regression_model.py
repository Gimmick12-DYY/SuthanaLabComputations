"""
This module contains the implementation of a logistic regression model for predicting LFP data from TRPTSD data.
This module is part of the TRPTSD project for the Suthana Lab Group.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



# Data Loading and Preprocessing