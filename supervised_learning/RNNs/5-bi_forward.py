#!/usr/bin/env python3
"""
    Module to create class BidirectionalCell that represents
    a bidirectional cell of an RNN
"""
import numpy as np


def softmax(x):
    """
        compute softmax activation
    """
    max_x = np.amax(x, 1).reshape(x.shape[0], 1)  # Get the row-wise maximum
    e_x = np.exp(x - max_x)  # For stability
    return e_x / e_x.sum(axis=1, keepdims=True)


def sigmoid(x):
    """
        compute sigmoid
    """
    return 1 / (1 + np.exp(-x))


class BidirectionalCell:
    """
        class for Bidirectional cell of an RNN
    """
    def __init__(self, i, h, o):
        """
            class constructor

        :param i: dimensionality of the data
        :param h: dimensionality of the hidden state
        :param o: dimensionality of the outputs
        """
        i_h_concat = i + h
        self.Whf = np.random.normal(size=(i_h_concat, h))
        self.Whb = np.random.normal(size=(i_h_concat, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bhf = np.zeros(shape=(1, h))
        self.bhb = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
            calculates the hidden state in the forward for one time step

        :param h_prev: ndarray, shape(m,h) previous hidden sate
        :param x_t: ndarray, shape(m,i) data input
            m: batch size

        :return: h_next
            n_next: next hidden state
        """
        # concat x_t and h_prev
        x_concat = np.concatenate((h_prev, x_t), axis=1)

        # forward direction
        h_forw = np.tanh(x_concat @ self.Whf + self.bhf)

        return h_forw
