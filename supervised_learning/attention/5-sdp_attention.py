#!/usr/bin/env python3
"""
    Scaled Dot-Product Attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
        calculates the scaled dot product attention

    :param Q: tensor, dim(...,seq_len_q,dk) query matrix
    :param K: tensor, dim(...,seq_len_v,dk) key matrix
    :param V: tensor, dim(...,seq_len_v,dv) value matrix
    :param mask: tensor, be broadcast into (..., seq_len_q,seq_len_v)
        optional mask or None
        if mask, multiply -1e9 to the mask and add to the scaled matrix

    :return: output, weights
        output: tensor, dim(...,seq_len_q,dv) scaled dot product attention
        weights: tensor, dim(...,seq_len_q,seq_len_v) attention weights
    """
    # use tensor flow to transpose
    dot_product = tf.matmul(Q, K, transpose_b=True)

    # extract d_k
    d_k = tf.cast(tf.shape(K)[-1], tf.float32)

    # scaling
    scaling = dot_product / tf.math.sqrt(d_k)

    # add mask
    if mask is not None:
        scaling += mask * -1e9

    # apply softmax to generate attention weight
    attention_weight = tf.nn.softmax(scaling, axis=-1)

    # weighted value
    output = tf.matmul(attention_weight, V)

    return output, attention_weight
