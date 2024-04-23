#!/usr/bin/env python3
"""
    Sparse autoencoder
"""
import tensorflow.keras as keras
import tensorflow as tf


def sampling(args):
    """
        sample new similar points from the latent space
    :param args: z_mean, z_log_sigma
    :param latent_dims: integer containing dimensions of the latent space
        representation
    :return: z_mean + exp(z_log_sigma) * epsilon
    """
    z_mean, z_log_sigma = args
    epsilon = keras.backend.random_normal(
        shape=(keras.backend.shape(z_mean)[0],
               keras.backend.shape(z_mean)[1]),
        mean=0.,
        stddev=0.1)

    return z_mean + keras.backend.exp(0.5 * z_log_sigma) * epsilon


def build_encoder(input_dims, hidden_layers, latent_dims):
    """
        built encoder part for a Vanilla autoencoder
    :param input_dims: integer containing dimensions of the model input
    :param hidden_layers: list containing number of nodes for each hidden
        layer in the encoder (should be reversed for the decoder)
    :param latent_dims: integer containing dimensions of the latent space
        representation
    :return: encoder model, mean, log variance
    """

    encoder_input = keras.layers.Input(shape=(input_dims,),
                                       name="encoder_input")
    encoder_layer = encoder_input
    for nodes in hidden_layers:
        encoder_layer = keras.layers.Dense(nodes,
                                           activation='relu'
                                           )(encoder_layer)
    z_mean = keras.layers.Dense(latent_dims,
                                activation=None,
                                name="mean")(encoder_layer)
    z_log_sigma = keras.layers.Dense(latent_dims,
                                     activation=None,
                                     name="log_variance"
                                     )(encoder_layer)

    z = (keras.layers.Lambda(lambda x: sampling(x))
         ([z_mean, z_log_sigma]))

    model_encoder = keras.Model(inputs=encoder_input,
                                outputs=[z_mean, z_log_sigma, z],
                                name="encoder")
    return model_encoder, z_mean, z_log_sigma


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
    decoder_input = keras.layers.Input(shape=(latent_dims,),
                                       name="decoder_input")
    decoder_layer = decoder_input

    for nodes in hidden_layers_decoder:
        decoder_layer = keras.layers.Dense(nodes,
                                           activation='relu'
                                           )(decoder_layer)
    decoder_output = keras.layers.Dense(output_dims,
                                        activation='sigmoid'
                                        )(decoder_layer)
    model_decoder = keras.Model(inputs=decoder_input,
                                outputs=decoder_output,
                                name="decoder")

    return model_decoder


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
        creates a sparse autoencoder
    :param input_dims: integer containing dimensions of the model input
    :param hidden_layers: list containing number of nodes for each hidden
        layer in the encoder (should be reversed for the decoder)
    :param latent_dims: integer containing dimensions of the latent space
        representation
    :return: encoder, decoder, auto
        encoder : encoder model, mean, log variance
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

    model_encoder, z_mean, z_log_sigma = (
        build_encoder(input_dims, hidden_layers, latent_dims))
    model_decoder = build_decoder(hidden_layers, latent_dims, input_dims)

    auto_input = model_encoder.input
    encoded_representation = model_encoder(auto_input)[0]
    decoded_representation = model_decoder(encoded_representation)

    autoencoder_model = keras.Model(inputs=auto_input,
                                    outputs=decoded_representation,
                                    name="vae_mlp")

    autoencoder_model.compile(loss='binary_crossentropy',
                              optimizer='adam')

    return model_encoder, model_decoder, autoencoder_model
