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
        self.Wy = np.random.normal(size=(h * 2, o))
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

    def backward(self, h_next, x_t):
        """
            calculates the hidden state in the backward direction
                for one time step

        :param h_next: ndarray, shape(m,h) next hidden state
        :param x_t: ndarray, shape(m,i) data input for the cell
            m: batch size

        :return: h_prev: previous hidden state
        """

        # concat x_t and h_next
        x_concat = np.concatenate((h_next, x_t), axis=1)

        # backward direction
        h_back = np.tanh(x_concat @ self.Whb + self.bhb)

        return h_back

    def output(self, H):
        """
            calculates all outputs for the RNN

        :param H: ndarray, shape(t,m,2*h) concatenated hidden states
            from both directions excluding initialized states
            t: number of time steps
            m: batch size for the data
            h: dimensionality of the hidden states

        :return: Y, outputs
        """
        t, m, two_h = H.shape

        Y = np.zeros((t, m, self.Wy.shape[1]))
        for step in range(t):
            # compute output
            Y[step] = H[step] @ self.Wy + self.by

            # softmax activation
            Y[step] = softmax(Y[step])

        return Y
