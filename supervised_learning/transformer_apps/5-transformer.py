#!/usr/bin/env python3
"""
    Transformer Architecture
"""
import tensorflow as tf
import numpy as np


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
        scaling += (mask * -1e9)

    # apply softmax to generate attention weight
    attention_weight = tf.nn.softmax(scaling, axis=-1)

    # weighted value
    output = tf.matmul(attention_weight, V)

    return output, attention_weight


def positional_encoding(max_seq_len, dm):
    """
        calculates the positional encoding for a transformer

    :param max_seq_len: integer, maximum sequence length
    :param dm: model depth

    :return: ndarray, shape(max_seq_len,dm)
        positional encoding vectors
    """
    # Initialize the positional encoding matrix with zeros
    PE = np.zeros((max_seq_len, dm))

    # Loop over each position in the sequence
    for i in range(max_seq_len):
        # Loop over each dimension of the positional encoding
        for j in range(0, dm, 2):
            # Compute the positional encoding using sin (even indices)
            PE[i, j] = np.sin(i / (10000 ** (j / dm)))
            # Compute the positional encoding using cos (odd indices)
            PE[i, j + 1] = np.cos(i / (10000 ** (j / dm)))

    # Return the positional encoding matrix
    return PE


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
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, shape=(batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, h, seq_len, dm)

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

        # split head and transpose
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_att, weights_att = sdp_attention(Q, K, V, mask)

        # transpose and reshape head
        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])
        concat_att = tf.reshape(scaled_att, (batch_size, -1, self.dm))

        # final projection
        output = self.linear(concat_att)

        return output, weights_att


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

        output_att2, weights_att2 = self.mha2(x_norm1,
                                              encoder_output,
                                              encoder_output,
                                              mask=padding_mask)
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

        return output, weights_att1, weights_att2


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
        output_att, weights_att = self.mha(Q, K, V, mask=mask)

        # dropout + Norm on residual connexion
        x_drop1 = self.dropout1(output_att, training=training)
        # residual connexion
        x = x + x_drop1
        x_norm1 = self.layernorm1(x)

        # Feed-forward network
        x_hidden = self.dense_hidden(x_norm1)
        x_output = self.dense_output(x_hidden)

        # dropout + Norm on residual connexion
        x_drop2 = self.dropout2(x_output, training=training)
        # residual connexion
        x = x_norm1 + x_drop2
        output = self.layernorm2(x)

        return output


class Decoder(tf.keras.layers.Layer):
    """
        classe for decoder of transformer
    """

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
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
        input_seq_len = x.shape[1]
        x = self.embedding(x)

        attention_weights = {}

        # positional encoding
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len]

        x = self.dropout(x, training=training)

        for i, block in enumerate(self.blocks):
            x, block1, block2 = block(x, encoder_output, training,
                                      look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        return x, attention_weights


class Encoder(tf.keras.layers.Layer):
    """
        class Encoder
    """

    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
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
        input_seq_len = x.shape[1]
        x = self.embedding(x)

        # positional encoding
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:input_seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x


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
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output, att_weights = self.decoder(target, enc_output, training, look_ahead_mask, decoder_mask)
        output = self.linear(dec_output)

        return output, att_weights
