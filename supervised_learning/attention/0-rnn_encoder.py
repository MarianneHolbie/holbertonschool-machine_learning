#!/usr/bin/env python3
"""
    Module to create Class RNN Encoder
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras as keras
import numpy as np


class RNNEncoder(Layer):
    """
        class to create RNN encoder for machine translation
    """
    def __init__(self, vocab, embedding, units, batch):
        """
            class constructor

        :param vocab: integer, size of input vocabulary
        :param embedding: integer, dimensionality of embedding vector
        :param units: integer, number hidden units in RNN cell
        :param batch: integer, batch size
        """
        if not all(isinstance(arg, int) for arg in [vocab, embedding, units, batch]):
            raise TypeError(f"{arg} Should be an integer.")

        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = keras.layers.Embedding(input_dim=vocab,
                                                output_dim=embedding)
        self.gru = keras.layers.GRU(units=units,
                                    return_sequences=True,
                                    return_state=True,
                                    bias_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        """
            initialize hidden states for RNN cell to a tensor of zeros

        :return: tensor, shape(batch, units), initialized hidden state
        """

        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
            function to call initial

        :param x: tensor, shape(batch, input_seq_len), input to the encoder layer
            as word indices within the vocabulary
        :param initial: tensor, shape(batch, units), initial hidden state

        :return: outputs, hidden
            outputs: tensor, shape(batch, input_seq_len, units)
                outputs of the encoder
            hidden: tensor, shape(batch, units)
                last hidden state of the encoder
        """

        x = self.embedding(x)

        # pass the embedding seq through the RNN
        outputs, hidden_state = self.gru(x, initial_state=initial)

        return outputs, hidden_state
