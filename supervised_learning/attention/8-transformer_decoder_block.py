#!/usr/bin/env python3
"""
    Class DecoderBlock
"""
import tensorflow as tf

MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
        class Decoder Block
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
            class constructor
            
        :param dm: dimensionality of the model
        :param h: number of heads
        :param hidden: number of hidden units in the fully connected layer
        :param drop_rate: dropout rate
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
            call method

        :param x: tensor, shape(batch,target_seq_len,dm)
            input to the decoder block
        :param encoder_output: tensor, shape(batch,input_seq_len,dm)
            output of the encoder
        :param training: bool, determine if model is training
        :param look_ahead_mask: mask for first multi head attention layer
        :param padding_mask: mask for second multi head attention layer

        :return: tensor, shape(batch,target_sep_len,dm)
            block's output
        """
        # same input x to generate Q, K and V
        Q = K = V = x

        # call MultiHeadAttention n1 layer with Q, K, V and mask
        output_att1, weights_att1 = self.mha1(Q, K, V, mask=look_ahead_mask)

        # dropout + Norm on residual connexion
        x_drop1 = self.dropout1(output_att1, training=training)
        # residual connexion
        x = x + x_drop1
        x_norm1 = self.layernorm1(x)

        Q = K = V = x_norm1
        # call MultiHeadAttention n2 layer with Q, K, V and mask
        output_att2, weights_att2 = self.mha2(Q, K, V, mask=padding_mask)
        # dropout + Norm on residual connexion
        x_drop2 = self.dropout2(output_att2, training=training)
        # residual connexion
        x = x_norm1 + x_drop2
        x_norm2 = self.layernorm2(x)

        hidden = self.dense_hidden(x_norm2)
        out = self.dense_output(hidden)
        x_drop3 = self.dropout3(out, training=training)
        # residual connexion
        x = x_norm2 + x_drop3
        output = self.layernorm3(x)

        return output
