#!/usr/bin/env python3
"""
    Forward propagation deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
        performs forward propagation for a deep RNN

    :param rnn_cells: list of RNNCell instances of length 1
        used for forward propagation
    :param X: ndarray, shape(t,m,i) data to be used
        t: maximum number of time steps
        m: batch size
        i: dimensionality of the data
    :param h_0: ndarray, shape(l,m,h), initial hidden state
        h: dimensionality of the hidden state

    :return: H, Y
        H: ndarray, all the hidden states
        Y: ndarray, all the outputs
    """
    if not isinstance(X, np.ndarray):
        raise TypeError('X should be a ndarray')
    if not isinstance(h_0, np.ndarray):
        raise TypeError('h_0 should be a ndarray')
    if X.shape[1] != h_0.shape[1]:
        raise ValueError("X.shape is (t,m,i) and h_0 shape is (l,m,h)")

    # extract dimension
    t, m, i = X.shape
    l, _, h = h_0.shape

    # initialization of H and Y
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = []

    # compute for each time step
    for i in range(t):
        # for each layer
        for layer in range(l):
            if layer == 0:
                h, y = rnn_cells[layer].forward(H[i][layer], X[i])
            else:
                h, y = rnn_cells[layer].forward(H[i][layer], h)

            H[i + 1][layer] = h
        Y.append(y)

    Y = np.array(Y)

    return H, Y
