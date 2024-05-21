#!/usr/bin/env python3
"""
    Positional encoding for a transformer
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
        calculates the positional encoding for a transformer

    :param max_seq_len: integer, maximum sequence length
    :param dm: model depth

    :return: ndarray, shape(max_seq_len,dm)
        positional encoding vectors
    """
    PE_vector = np.zeros(shape=(max_seq_len, dm))

    for pos in range(max_seq_len):
        for i in range(dm):
            if pos % 2 == 0:
                PE_vector[pos, i] = np.sin(pos / (10000 ** (i / dm)))
            else:
                PE_vector[pos, i] = np.cos(pos / (10000 ** ((i - 1) / dm)))

    return PE_vector
