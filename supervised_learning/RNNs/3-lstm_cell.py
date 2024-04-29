#!/usr/bin/env python3
"""
    Module to create class LSTMCell that represents a LSTM unit
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


class LSTMCell:
    """
        class LSTM unit
    """
    def __init__(self, i, h, o):
        """
            class constructor

        :param i: dimensionality of the data
        :param h: dimensionality of the hidden state
        :param o: dimensionality of the outputs
        """
        i_h_concat = i + h
        self.Wf = np.random.normal(size=(i_h_concat, h))
        self.Wu = np.random.normal(size=(i_h_concat, h))
        self.Wc = np.random.normal(size=(i_h_concat, h))
        self.Wo = np.random.normal(size=(i_h_concat, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros(shape=(1, h))
        self.bu = np.zeros(shape=(1, h))
        self.bc = np.zeros(shape=(1, h))
        self.bo = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
            performs forward propagation for one time step

        :param h_prev: ndarray, shape(m,h) previous hidden state
        :param c_prev: ndarray, shape(m,h) previous cell state
        :param x_t: ndarray, shape(m,i) data input
                m: batch size for the data
        output: softmax activation

        :return: h_next, c_next, y
            h_next: next hidden state
            c_next: newt cell state
            y: output of the cell
        """

        # concat x_t and h_prev
        x_concat = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        f_x = sigmoid(x_concat @ self.Wf + self.bf)

        # update gate
        u_x = sigmoid(x_concat @ self.Wu + self.bu)

        # new information cell state
        c_x = np.tanh(x_concat @ self.Wc + self.bc)

        # new cell state
        input_gate = c_x * u_x
        c_next = f_x * c_prev + input_gate

        # output gate
        o_t = sigmoid(x_concat @ self.Wo + self.bo)

        # outputs
        h_next = o_t * np.tanh(c_next)

        y = softmax(h_next @ self.Wy + self.by)

        return h_next, c_next, y
