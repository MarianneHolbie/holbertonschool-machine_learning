#!/usr/bin/env python3
from tensorflow import keras


def convolutional_GenDiscr():
    """
        function to create generator and discriminator for GAN model
    :return:
    """

    # generator model
    def conv_block_g(x,
                     filters,
                     kernel_size,
                     strides=(1, 1),
                     up_size=(2, 2),
                     padding='same'
                     ):
        """
            function to create conv block in generator
        :param x: input
        :param filters: int, number of filter
        :param kernel_size: int, kernel size
        :param strides: tuple, dor stride
        :param up_size: tuple, size for upSampling2D layer
        :param padding: string, name of padding's type
        :return:
        """
        x = keras.layers.UpSampling2D(up_size)(x)
        x = keras.layers.Conv2D(filters,
                                kernel_size,
                                strides,
                                padding,
                                activation='tanh')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('tanh')(x)
        return x

    def get_generator():
        """
            function to construct the generator
        :return: generator model
        """
        inputs = keras.Input(shape=(16,))
        hidden = keras.layers.Dense(2048, activation='tanh')(inputs)
        x = keras.layers.Reshape((2, 2, 512))(hidden)
        x = conv_block_g(x, 64, (3, 3), (1, 1))
        x = conv_block_g(x, 16, (3, 3))
        outputs = conv_block_g(x, 1, (3, 3))
        generator = keras.Model(inputs, outputs, name="generator")
        return generator

    # discriminator model
    def conv_block_d(x,
                     filters,
                     kernel_size,
                     strides=(2, 2),
                     padding='same',
                     pool_size=(2, 2)):
        """
            conv block for discriminator
        :param x: input
        :param filters: int, number of filter
        :param kernel_size: int, size of kernel
        :param strides: typle, value for stride
        :param padding: string, name of padding's type
        :param pool_size: tuple, size for Maxpooling2D layer
        :return:
        """
        x = keras.layers.Conv2D(filters,
                                kernel_size,
                                (1, 1),
                                padding)(x)
        x = keras.layers.MaxPooling2D(pool_size, strides, padding)(x)
        x = keras.layers.Activation('tanh')(x)
        return x

    def get_discriminator():
        """
            construct discriminator model
        :return: discriminator model
        """
        inputs = keras.Input(shape=(16, 16, 1))
        x = conv_block_d(inputs,
                         32,
                         (3, 3))
        x = conv_block_d(x,
                         64,
                         (3, 3))
        x = conv_block_d(x,
                         128,
                         (3, 3))
        x = conv_block_d(x,
                         256,
                         (3, 3))

        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation='tanh')(x)
        discriminator = keras.Model(inputs, outputs, name="discriminator")
        return discriminator

    return get_generator(), get_discriminator()
