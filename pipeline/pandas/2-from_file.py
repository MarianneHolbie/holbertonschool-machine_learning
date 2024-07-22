#!/usr/bin/env python3
"""
From File
"""
import pandas as pd


def from_file(filename, delimiter):
    """
        loads data from a file as a pd.DataFrame

    :param filename: file to load from
    :param delimiter: column separator

    :return: pd.DataFrame
    """
    df = pd.read_csv(filename,
                     delimiter=delimiter)

    return df
