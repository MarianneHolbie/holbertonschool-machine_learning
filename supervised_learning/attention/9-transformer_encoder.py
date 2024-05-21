#!/usr/bin/env python3
"""
    Encoder Transformer
"""
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
        class Encoder
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
            class initialization
        :param N: number of blocks in the encoder
        :param dm: dimensionality of the model
        :param h: number of heads
        :param hidden: number of hidden units in the fully connected layer
        :param input_vocab: size of the input vocabulary
        :param max_seq_len: max sequence length possible
        :param drop_rate: dropout rate
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=self.dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = []
        for i in range(N):
            self.blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
            call method
        :param x: tensor, shape(batch, input_seq_len, dm)
            input to the encoder
        :param training: bool, determine if model is training
        :param mask: for multihead attention

        :return: tensor, shape(batch,input_seq_len, dm)
            encoder output
        """
        x = self.embedding(x)
        input_seq_len = tf.shape(x)[1]

        # positional encoding
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len, :]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x
