#!/usr/bin/env python3
"""
    Class EncoderBlock
"""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
        class encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
            class constructor

        :param dm: dimensionality model
        :param h: number of head
        :param hidden: number of hidden units in the fully
            connected layer
        :param drop_rate: dropout rate
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
            call method

        :param x: tensor, shape(batch,input_seq_len,dm)
            input to the encoder block
        :param training: boolean to determine if the model is training
        :param mask: mask to applied for multi head attention

        :return: tensor, shape(batch,input_seq_len,dm)
            block's output
        """
        # same input x to generate Q, K and V
        Q = K = V = x

        # call MultiHeadAttention layer with Q, K, V and mask
        output, weights_att = self.mha(Q, K, V, mask=mask)

        # residual connexion
        x = x + output

        # Layer normalization + dropout
        x_norm1 = self.layernorm1(x)
        x_drop1 = self.dropout1(x_norm1, training=training)

        # Feed-forward network
        x_hidden = self.dense_hidden(x_drop1)
        x_output = self.dense_output(x_hidden)

        # residual connexion
        x = x_output + x_drop1

        # Layer normalization + dropout
        x_norm2 = self.layernorm2(x)
        output = self.dropout2(x_norm2, training=training)

        return output
