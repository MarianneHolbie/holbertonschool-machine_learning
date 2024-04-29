#!/usr/bin/env python3
"""
    Module to create class GRU Cell that represents a gated recurrent unit
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


class GRUCell:
    """
        class represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
            class constructor

        :param i: dimensionality of the date
        :param h: dimensionality of the hidden state
        :param o: dimensionality of the outputs
        """
        i_h_concat = i + h
        self.Wz = np.random.normal(size=(i_h_concat, h))
        self.Wr = np.random.normal(size=(i_h_concat, h))
        self.Wh = np.random.normal(size=(i_h_concat, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros(shape=(1, h))
        self.bz = np.zeros(shape=(1, h))
        self.br = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
            performs forward propagation for one time step

        :param h_prev: ndarray, shape(m,h) previous hidden state
        :param x_t: ndarray, shape(m,i) data input
                m: batch size for the data
        output: softmax activation

        :return: h_next, y
            h_next: next hidden state
            y: output of the cell
        """

        # concat x_t and h_prev
        x_concat = np.concatenate((h_prev, x_t), axis=1)

        # update gate
        z_t = sigmoid(x_concat @ self.Wz + self.bz)

        # reset gate
        r_t = sigmoid(x_concat @ self.Wr + self.br)

        # intermediate hidden state with tanh activation
        h_intermediate = np.tanh(
            np.concatenate((r_t * h_prev, x_t), axis=1) @ self.Wh + self.bh)

        h_next = z_t * h_intermediate + (1 - z_t) * h_prev

        # output with softmax activation
        y = softmax(h_next @ self.Wy + self.by)

        return h_next, y
