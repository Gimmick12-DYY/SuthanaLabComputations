"""
Usage of LSTM analyisis for TR-PTSD data.
This is another model attempt for analyzing the time-series data provdided.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def load_data (data_path):
    data = np.load(data_path, allow_pickle=True)
    return data

