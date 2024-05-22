#!/usr/bin/env python3
"""
    Transformer
"""
import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """
        class Transformer
    """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
            class init

        :param N: number blocks in encoder & decoder
        :param dm: dimensionality of the model
        :param h: number of heads
        :param hidden: number hidden units in fully connected layer
        :param input_vocab: size input vocab
        :param target_vocab: size target vocab
        :param max_seq_input: max seq length for input
        :param max_seq_target: max seq length for target
        :param drop_rate: dropout rate
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
            call function

        :param inputs: tensor, shape(batch,input_seq_len) inputs
        :param target: tensor, shape(batch,target_seq_len) target
        :param training: bool, determine if model training
        :param encoder_mask: mask apply to encoder
        :param look_ahead_mask: mask apply to decoder
        :param decoder_mask: padding mask apply to the decoder

        :return: tensor, shape(batch,target_seq_len,target_vocab)
            transformer output
        """
        x = self.encoder(inputs, training, encoder_mask)
        x = self.decoder(target, x, training, look_ahead_mask, decoder_mask)
        output = self.linear(x)

        return output
