#!/usr/bin/env python3
"""
    Convolutional autoencoder
"""
import tensorflow.keras as keras


def build_encoder(input_dims, filter, latent_dims):
    """
        built encoder part for a Vanilla autoencoder

    :param input_dims: tuple of integer containing dimensions of the
        model input
    :param filter: list containing number of filter for each conv
        layer in the encoder (should be reversed for the decoder)
    :param latent_dims: tuple of int containing dimensions of the
        latent space representation

    :return: encoder model
    """

    encoder_input = keras.layers.Input(shape=input_dims,
                                       name="encoder_input")
    encoder_layer = encoder_input
    for f in filter:
        encoder_layer = keras.layers.Conv2D(f,
                                            (3, 3),
                                            activation='relu',
                                            padding='same')(encoder_layer)
        encoder_layer = keras.layers.MaxPooling2D((2, 2),
                                                  padding='same'
                                                  )(encoder_layer)
    model_encoder = keras.Model(inputs=encoder_input,
                                outputs=encoder_layer,
                                name="encoder")
    return model_encoder


def build_decoder(filter, latent_dims, output_dims):
    """
        build decoder part for a Vanilla Autoencoder

    :param filter: list containing number of filter for each conv
        layer in the encoder (should be reversed for the decoder)
    :param latent_dims: tuple of int containing dimensions of the
        latent space representation
    :param output_dims: tuple of int containing dimensions output

    :return: decoder model
    """
    filter_layers_decoder = list(reversed(filter[1:]))
    decoder_input = keras.layers.Input(shape=latent_dims,
                                       name="decoder_input")
    decoder_layer = decoder_input

    for f in filter_layers_decoder:
        decoder_layer = keras.layers.Conv2D(f,
                                            (3, 3),
                                            strides=(1, 1),
                                            activation='relu',
                                            padding='same')(decoder_layer)
        decoder_layer = keras.layers.UpSampling2D((2, 2))(decoder_layer)
    decoder_layer = keras.layers.Conv2D(filter[0],
                                        (3, 3),
                                        padding='valid',
                                        strides=(1, 1),
                                        activation='relu')(decoder_layer)
    decoder_layer = keras.layers.UpSampling2D((2, 2))(decoder_layer)
    decoder_output = keras.layers.Conv2D(output_dims[-1],
                                         (3, 3),
                                         activation='sigmoid',
                                         strides=(1, 1),
                                         padding='same'
                                         )(decoder_layer)

    model_decoder = keras.Model(inputs=decoder_input,
                                outputs=decoder_output,
                                name="decoder")

    return model_decoder


def autoencoder(input_dims, filter, latent_dims):
    """
        creates a sparse autoencoder

    :param input_dims: tuple of int containing dimensions of the model input
    :param filter: list containing number of filters for each conv
        layer in the encoder (should be reversed for the decoder)
    :param latent_dims: tuple of integer containing dimensions of the
        latent space representation

    :return: encoder, decoder, auto
        encoder : encoder model
        decoder: decoder model
        auto: full autoencoder model

        compilation : Adam opt, binary cross-entropy loss
        layer: relu activation except last layer decoder : sigmoid
    """

    if not isinstance(input_dims, tuple):
        raise TypeError("input_dims should be an integer")
    if not isinstance(latent_dims, tuple):
        raise TypeError("input_dims should be an integer")
    if not isinstance(filter, list):
        raise TypeError("filter should be a list")

    model_encoder = build_encoder(input_dims, filter, latent_dims)
    model_decoder = build_decoder(filter, latent_dims, input_dims)

    auto_input = model_encoder.input
    encoded_representation = model_encoder(auto_input)
    decoded_representation = model_decoder(encoded_representation)

    autoencoder_model = keras.Model(inputs=auto_input,
                                    outputs=decoded_representation)

    autoencoder_model.compile(loss='binary_crossentropy',
                              optimizer='adam')

    return model_encoder, model_decoder, autoencoder_model
