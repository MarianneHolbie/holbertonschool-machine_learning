#!/usr/bin/env python3
"""
    "Vanilla" autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
        function that creates an autoencoder

    :param input_dims: integer containing dimensions of the model input
    :param hidden_layers: list containing number of nodes for each hidden layer in the
        encoder (should be reversed for the decoder)
    :param latent_dims: integer containing dimensions of the latent space representation

    :return: encoder, decoder, auto
        encoder : encoder model
        decoder: decoder model
        auto: full autoencoder model

        compilation : Adam opt, binary cross-entropy loss
        layer: relu activation except last layer decoder : sigmoid
    """

    if not isinstance(input_dims, int):
        raise TypeError("input_dims should be an integer")
    if not isinstance(latent_dims, int):
        raise TypeError("input_dims should be an integer")
    if not isinstance(hidden_layers, list):
        raise TypeError("hidden_layers should be a list")

    model_encoder = keras.Sequential()

    for i in range(len(hidden_layers)):
        model_encoder.add(keras.layers.Dense(hidden_layers[i],
                                             activation='relu'))
    model_encoder.add(keras.layers.Dense(latent_dims,
                                         activation='relu'))

    model_decoder = keras.Sequential()

    for i in reversed(range(len(hidden_layers))):
        model_decoder.add(keras.layers.Dense(hidden_layers[i],
                                             activation='relu'))
    model_decoder.add(keras.layers.Dense(input_dims,
                                         activation='sigmoid'))

    inputs = keras.layers.Input(shape=(input_dims,))
    encoded_representation = model_encoder(inputs)
    decoded_representation = model_decoder(encoded_representation)

    autoencoder_model = keras.Model(inputs=inputs, outputs=decoded_representation)

    autoencoder_model.compile(loss=keras.losses.BinaryCrossentropy(),
                              optimizer=keras.optimizers.Adam())

    return model_encoder, model_decoder, autoencoder_model
