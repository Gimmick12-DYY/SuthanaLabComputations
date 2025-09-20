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


def get_batch(data, indices):
    batch_data = [data[i] for i in indices]
    sequences = [item['sequence'] for item in batch_data]
    labels = [item['label'] for item in batch_data]
    lengths = [len(seq) for seq in sequences]

    padded_sequences = pad_sequences(sequences, padding='post', value=0.0)
    labels = np.array(labels)

    return padded_sequences, labels, lengths

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # Pack the sequences
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        # Unpack the sequences
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last time step output for each sequence
        idx = (torch.tensor(lengths) - 1).view(-1, 1).expand(len(lengths), output.size(2)).unsqueeze(1)
        last_output = output.gather(1, idx).squeeze(1)
        out = self.dropout(last_output)
        out = self.fc(out)
        return out