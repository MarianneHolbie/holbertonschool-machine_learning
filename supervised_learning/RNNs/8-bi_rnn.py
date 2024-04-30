#!/usr/bin/env python3
"""
    Forward propagation for bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
        performs forward propagation for bidirectional RNN

    :param bi_cell: instance of BidirectionalCell
    :param X: ndarray, shape(t,m,i) data to be used
        t: maximum number of time steps
        m: batch size
        i: dimensionality of the data
    :param h_0: ndarray, shape(m,h) initial hidden state in forward direction
        h: dimensionality of the hidden state
    :param h_t: ndarray, shape(m,h) initial hidden state in backward direction

    :return: H, Y
        H: ndarray, all the concatenated hidden states
        Y: all the outputs
    """
    if not isinstance(X, np.ndarray):
        raise TypeError('X should be a ndarray')
    if not isinstance(h_0, np.ndarray):
        raise TypeError('h_0 should be a ndarray')
    if not isinstance(h_t, np.ndarray):
        raise TypeError('h_t should be a ndarray')
    if X.shape[1] != h_0.shape[0] or X.shape[1] != h_t.shape[0]:
        raise ValueError('Verify your dimension')

    # extract dimension
    t, m, i = X.shape
    _, h = h_0.shape

    # initialization of H and Y
    H_forw = np.zeros((t + 1, m, h))
    H_forw[0] = h_0
    H_back = np.zeros((t + 1, m, h))
    H_back[-1] = h_t
    Y = []

    for step in range(t):
        h_f = bi_cell.forward(H_forw[step], X[step, :, :])
        H_forw[step + 1] = h_f

    for step in reversed(range(t)):
        h_b = bi_cell.backward(H_back[step + 1], X[step, :, :])
        H_back[step] = h_b

    # output
    H = np.concatenate((H_forw[1:], H_back[:-1]), axis=2)
    Y = bi_cell.output(H)

    return H, Y
