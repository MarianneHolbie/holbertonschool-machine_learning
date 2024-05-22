#!/usr/bin/env python3
"""
    Decoder Transformer
"""
import tensorflow as tf

positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
        classe for decoder of transformer
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """
            class init
        :param N: number of blocks in the encoder
        :param dm: dimensionality of the model
        :param h: number of heads
        :param hidden: number of hidden units in the fully connected layer
        :param target_vocab: size of the target vocabulary
        :param max_seq_len: max sequence length possible
        :param drop_rate: dropout rate
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=self.dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = []
        for i in range(N):
            self.blocks.append(DecoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
            call function
        :param x: tensor, shape(batch,target_seq_len,dm)
            input to the decoder
        :param encoder_output: tensor, shape(batch,input_seq_len,dm)
            output of the decoder
        :param training: bool, define if model is training
        :param look_ahead_mask: mask for first multihead layer
        :param padding_mask: mask for second multihead layer

        :return:tensor, shape(batch,target_seq_len,dm)
            decoder output
        """
        x = self.embedding(x)
        input_seq_len = tf.shape(x)[1]

        # positional encoding
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len, :]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask, padding_mask)

        return x
