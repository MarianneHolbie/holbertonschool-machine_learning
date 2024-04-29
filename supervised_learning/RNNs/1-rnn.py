#!/usr/bin/env python3
"""
    Forward propagation simple RNN
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
        performs forward propagation for a simple RNN

    :param rnn_cell: instance of RNNCell
    :param X: ndarray, shape(t,m,i) data to be used
        t: maximum number of time steps
        m: batch size
        i: dimensionality of the data
    :param h_0: ndarray, shape(m,h), initial hidden state
        h: dimensionality of the hidden state

    :return: H, Y
        H: ndarray, all the hidden states
        Y: ndarray, all the outputs
    """
    if not isinstance(X, np.ndarray):
        raise TypeError('X should be a ndarray')
    if not isinstance(h_0, np.ndarray):
        raise TypeError('h_0 should be a ndarray')
    if X.shape[1] != h_0.shape[0]:
        raise ValueError("X.shape is (t,m,i) and h_0 shape is (m,h)")

    # extract dimension
    t, m, i = X.shape
    _, h = h_0.shape

    # initialization of h_prev and y
    h_prev = h_0
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    # compute for each time step
    for i in range(t):
        # use function forward form class rnn_cell
        h_next, y = rnn_cell.forward(h_prev, X[i])
        h_prev = h_next
        H[i + 1] = h_next
        Y[i] = y

    return H, Y
