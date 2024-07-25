#!/usr/bin/env python3
"""
Crop
"""
import tensorflow as tf


def crop_image(image, size):
    """
        performs a random crop of an image

    :param image: tf.Tensor, image to crop
    :param size: size of the crop

    :return: cropped image
    """
    return tf.image.random_crop(image, size)
