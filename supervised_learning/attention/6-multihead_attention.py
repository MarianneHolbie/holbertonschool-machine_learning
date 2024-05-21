#!/usr/bin/env python3
"""
    Class Multi Head Attention
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
        class to perform multi head attention
    """

    def __init__(self, dm, h):
        """
            class constructor

        :param dm: int, dimensionality of the model
        :param h: int, number of heads
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=self.dm)
        self.Wk = tf.keras.layers.Dense(units=self.dm)
        self.Wv = tf.keras.layers.Dense(units=self.dm)
        self.linear = tf.keras.layers.Dense(units=self.dm)

    def call(self, Q, K, V, mask):
        """
            call method

        :param Q: tensor, shape(batch,seq_len_q,dk)
            input to generate the query matrix
        :param K: tensor, shape(batch,seq_len_v,dk)
            input to generate the key matrix
        :param V: tensor, shape(batch,seq_len_v,dv)
            input to generate the value matrix
        :param mask: always None

        :return: output, weights
            output: tensor, dim(...,seq_len_q,dm)
                scaled dot product attention
            weights: tensor, dim(...,h,seq_len_q,seq_len_v)
                attention weights
        """
        # extract batch size
        batch_size = tf.shape(Q)[0]

        # Request matrix projection
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # split and transpose
        Q = tf.reshape(Q, (batch_size, -1, self.h, self.depth))
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.reshape(K, (batch_size, -1, self.h, self.depth))
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.reshape(V, (batch_size, -1, self.h, self.depth))
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        scaled_att, weights_att = sdp_attention(Q, K, V, mask)

        # transpose and reshape head
        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])
        concat_att = tf.reshape(scaled_att, (batch_size, -1, self.dm))

        # final projection
        output = self.linear(concat_att)

        return output, weights_att
