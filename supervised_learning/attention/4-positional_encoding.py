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
        for i in np.arange(int(dm / 2)):
            denominator = 10000 ** (2 * i / dm)
            PE_vector[pos, 2 * i] = np.sin(pos / denominator)
            PE_vector[pos, 2 * i + 1] = np.cos(pos/denominator)

    return PE_vector
