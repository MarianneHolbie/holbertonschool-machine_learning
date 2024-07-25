#!/usr/bin/env python3
"""
Shear
"""
from tensorflow import keras as k


def shear_image(image, intensity):
    """
        randomly shears an image

    :param image: tf.Tensor, image to shear
    :param intensity: intensity with the image should be sheared

    :return: sheared image
    """
    # convert image to numpy arrays
    image_np = image.numpy()
    return k.preprocessing.image.random_shear(image_np,
                                              intensity,
                                              row_axis=0,
                                              col_axis=1,
                                              channel_axis=2)
