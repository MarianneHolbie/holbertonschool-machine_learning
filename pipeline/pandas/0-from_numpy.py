#!/usr/bin/env python3
"""
From Numpy
"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """
        function that creates a pd.DataFrame from a np.ndarray

    :param array: ndarray, from which you should create the DataFrame

    :return: DataFrame,
        column labeled in alphabetical order and capitalized
    """
    # verification
    if not isinstance(array, np.ndarray):
        raise TypeError("Input should be an np.ndarray")

    # determine number of columns
    num_col = array.shape[1]

    # create name of columns
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    columns = alphabet[:num_col]

    # create dataframe
    df = pd.DataFrame(array,
                      columns=columns)

    return df
