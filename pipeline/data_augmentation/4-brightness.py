#!/usr/bin/env python3
"""
Brightness
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
        randomly changes the brightness of an image

    :param image: tf.tensor, image
    :param max_delta: max amount

    :return: altered image
    """

    return tf.image.adjust_brightness(image, max_delta)
