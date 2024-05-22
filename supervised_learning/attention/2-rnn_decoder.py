#!/usr/bin/env python3
"""
    Module to create Class RNN Decoder
"""
import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
        class to create RNN decoder for machine translation
    """

    def __init__(self, vocab, embedding, units, batch):
        """
            class constructor

        :param vocab: integer, size of output vocabulary
        :param embedding: integer, dimensionality of embedding vector
        :param units: integer, number hidden units in RNN cell
        :param batch: integer, batch size
        """
        invalid_args = [arg for arg in [vocab, embedding, units, batch]
                        if not isinstance(arg, int)]
        if invalid_args:
            arg_str = ", ".join([f"{arg}" for arg in invalid_args])
            raise TypeError(f"{arg_str} Should be an integer.")

        super().__init__()
        self.units = units
        self.batch = batch
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """
            call function

        :param x: tensor, shape(batch,1), previous word in the target
        :param s_prev: tensor, shape(batch, units) previous decoder
            hidden state
        :param hidden_states: tensor, shape(batch,input_seq_len,units)
             outputs of the encoder

        :return: y, s
            y: tensor, shape(batch, vocab) output word as a one hot
                vector in the target vocabulary
            s: tensor, shape(batch, units) new decoder hidden state
        """
        # embedding vector
        x = self.embedding(x)  # shape(32, 1, 128)

        # concat embedding vector with previous hidden state
        s_prev = tf.expand_dims(s_prev, 1)  # shape(32, 1, 256)
        x = tf.concat([s_prev, x], axis=-1)  # shape(32, 1, 256 + 128)

        # context and weigh
        # output shape(32, 10, 256)
        context, att_weights = self.attention(x, hidden_states)
        print(context.shape)

        # concatenate context with embedding vector
        # reshape in shape (32, 10, units + embedding_dim)
        x = tf.tile(x, [1, context.shape[1], 1])
        # shape output: (32, 10, units + units + embedding_dim)
        context_concat = tf.concat([context, x], axis=-1)

        outputs, hidden_state = self.gru(context_concat)

        # supress dim=1
        new_outputs = tf.reshape(outputs, (-1, outputs.shape[2]))

        y = self.F(new_outputs)

        return y, hidden_state
