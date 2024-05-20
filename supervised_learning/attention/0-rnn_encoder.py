#!/usr/bin/env python3
"""
    Module to create Class RNN Encoder
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
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
        invalid_args = [arg for arg in [vocab, embedding, units, batch]
                        if not isinstance(arg, int)]
        if invalid_args:
            arg_str = ", ".join([f"{arg}" for arg in invalid_args])
            raise TypeError(f"{arg_str} Should be an integer.")

        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer=tf.keras.initializers.GlorotUniform())

    def initialize_hidden_state(self):
        """
            initialize hidden states for RNN cell to a tensor of zeros

        :return: tensor, shape(batch, units), initialized hidden state
        """

        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """
            function to call initial

        :param x: tensor, shape(batch, input_seq_len), input to the
            encoder layer as word indices within the vocabulary
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
