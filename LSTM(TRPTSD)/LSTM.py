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

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.):
    """Pads sequences to the same length.

    Args:
        sequences (list of list): List of sequences to be padded.
        maxlen (int, optional): Maximum length of all sequences. If None, uses the length of the longest sequence.
        dtype (str, optional): Data type of the output array.
        padding (str, optional): 'pre' or 'post', pad either before or after each sequence.
        truncating (str, optional): 'pre' or 'post', remove values from sequences larger than maxlen either in the beginning or in the end of the sequence.
        value (float, optional): Value to pad with.

    Returns:
        numpy.ndarray: Padded sequences array.
    """
    lengths = [len(s) for s in sequences]
    if maxlen is None:
        maxlen = max(lengths)

    padded_sequences = np.full((len(sequences), maxlen) + np.shape(sequences[0])[1:], value, dtype=dtype)

    for i, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        else:
            trunc = s[:maxlen]

        if padding == 'post':
            padded_sequences[i, :len(trunc)] = trunc
        else:
            padded_sequences[i, -len(trunc):] = trunc

    return padded_sequences
