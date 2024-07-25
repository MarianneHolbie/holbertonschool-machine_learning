#!/usr/bin/env python3
"""
Rotate
"""
import tensorflow as tf


def rotate_image(image):
    """
        rotates an image by 90 degrees counter-clockwise

    :param image: tf.Tensor, image to rotate

    :return: rotated image
    """
    return tf.image.rot90(image, k=1)
