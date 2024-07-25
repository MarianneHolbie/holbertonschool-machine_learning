#!/usr/bin/env python3
"""
Flip
"""
import tensorflow as tf


def flip_image(image):
    """
        flips an image horizontally

    :param image: tf.Tensor, image to flip

    :return: flipped image
    """
    return tf.image.flip_left_right(image)
