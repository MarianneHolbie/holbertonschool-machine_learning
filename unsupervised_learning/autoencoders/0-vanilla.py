#!/usr/bin/env python3
"""
    "Vanilla" autoencoder
"""
import tensorflow.keras as keras


def build_encoder(input_dims, hidden_layers, latent_dims):
    """
        built encoder part for a Vanilla autoencoder

    :param input_dims: integer containing dimensions of the model input
    :param hidden_layers: list containing number of nodes for each hidden
        layer in the encoder (should be reversed for the decoder)
    :param latent_dims: integer containing dimensions of the latent space
        representation

    :return: encoder model
    """
    inputs = keras.layers.Input(shape=(input_dims,))
    x = inputs
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    encoded = keras.layers.Dense(latent_dims, activation='relu')(x)
    model = keras.Model(inputs=inputs, outputs=encoded)
    return model


def build_decoder(hidden_layers, latent_dims, output_dims):
    """
        build decoder part for a Vanilla Autoencoder

    :param hidden_layers: list containing number of nodes for each hidden
        layer in the encoder (should be reversed for the decoder)
    :param latent_dims: integer containing dimensions of the latent space
        representation
    :param output_dims: integer containing dimensions output

    :return: decoder model
    """
    hidden_layers_decoder = list(reversed(hidden_layers))
    inputs = keras.layers.Input(shape=(latent_dims,))
    x = inputs

    for nodes in hidden_layers_decoder:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoded = keras.layers.Dense(output_dims, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=decoded)

    return model


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
        function that creates an autoencoder

    :param input_dims: integer containing dimensions of the model input
    :param hidden_layers: list containing number of nodes for each hidden
        layer in the encoder (should be reversed for the decoder)
    :param latent_dims: integer containing dimensions of the latent space
        representation

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

    model_encoder = build_encoder(input_dims, hidden_layers, latent_dims)
    model_decoder = build_decoder(hidden_layers, latent_dims, input_dims)

    inputs = model_encoder.inputs
    encoded_representation = model_encoder.outputs
    decoded_representation = model_decoder(encoded_representation)

    autoencoder_model = keras.Model(inputs=inputs,
                                    outputs=decoded_representation)

    autoencoder_model.compile(loss='binary_crossentropy',
                              optimizer='adam')

    return model_encoder, model_decoder, autoencoder_model
