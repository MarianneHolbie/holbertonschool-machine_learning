#!/usr/bin/env python3
"""
    Function create_masks
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
        creates all masks for training/validation

    :param inputs: tf.tensor, shape(batch_size,seq_len_in)
        input sequence
    :param target: tf.tensor, shape(batch_size,seq_len_out)
        target sentence

    :return:encoder_mask, combined_mask, decoder_mask
        encoder_mask: tf.tensor, padding mask shape(batch_size,1,1,seq_len_in)
            apply in the encoder
        combined_mask: tf.tensor, shape(batch_size,1,len_seq_out,seq_len_out)
            apply in first attention block in decoder to pad
            and mask future tokens in the input received by decoder
        decoder_mask: tf.tensor padding mask, shape(batch_size,1,1,seq_len_in)
            apply 2nd attention block in the decoder
    """

    def create_padding_mask(x):
        """
            create a padding mask on sequence

        :param x: sequence
        :return: corresponding mask
            shape(batch_size,1,1,seq_len_in)
        """
        # Create mask which marks the zero padding values in the input by a 1
        mask_p = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask_p = mask_p[:, tf.newaxis, tf.newaxis, :]
        return mask_p

    def create_look_ahead_mask(size):
        """
            create look ahead mask

        :param size: size of the seq
        :return: mask shape(seq_len, seq_len)
        """
        # Mask out future entries by marking them with a 1.0
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    # Extract batch size and sequence lengths from inputs and target tensors
    batch_size, seq_len_in = inputs.shape
    seq_len_out = target.shape[1]

    # padding mask
    encoder_mask = create_padding_mask(inputs)
    encoder_mask = tf.reshape(encoder_mask,
                              shape=(batch_size, 1, 1, seq_len_in))
    decoder_target_mask = create_padding_mask(target)
    decoder_target_mask = tf.reshape(decoder_target_mask,
                                     shape=(batch_size, 1, 1, seq_len_out))

    # look Ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len_out)

    # combined mask
    combined_mask = tf.maximum(decoder_target_mask, look_ahead_mask)

    # decoder mask
    decoder_mask = create_padding_mask(inputs)
    decoder_mask = tf.reshape(decoder_mask,
                              shape=(batch_size, 1, 1, seq_len_in))

    return encoder_mask, combined_mask, decoder_mask
