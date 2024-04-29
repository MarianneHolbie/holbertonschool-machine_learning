#!/usr/bin/env python3
"""
    Module to create class RNNCell that represents a cell of a simple RNN
"""
import numpy as np


def softmax(x):
    """
        compute softmax activation
    """
    max_x = np.amax(x, 1).reshape(x.shape[0], 1)  # Get the row-wise maximum
    e_x = np.exp(x - max_x)  # For stability
    return e_x / e_x.sum(axis=1, keepdims=True)


class RNNCell:
    """
        class that represent a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
            class constructor

        :param i: dimensionality of the data
        :param h: dimensionality of the hidden state
        :param o: dimensionality of the outputs
        """
        i_h_concat = i + h
        self.Wh = np.random.normal(size=(i_h_concat, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
            performs forward propagation for one time step

        :param h_prev: ndarray, shape(m,h) previous hidden state
        :param x_t: ndarray, shape(m,i) data input for cell
            m: batche size for the data

        :return: h_next, y
            h_next: next hidden state
            y: output of the cell
        """

        # concat x_t and h_prev
        x_concat = np.concatenate((h_prev, x_t), axis=1)

        # next hidden state with tanh activation
        h_next = np.tanh((x_concat @ self.Wh + self.bh))

        # output with softmax activation
        y = softmax(h_next @ self.Wy + self.by)

        return h_next, y
