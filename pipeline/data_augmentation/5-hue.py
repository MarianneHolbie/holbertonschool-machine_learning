#!/usr/bin/env python3
"""
Hue
"""
import tensorflow as tf


def change_hue(image, delta):
    """
        changes the hue of an image

    :param image: tf.tensor, image
    :param delta: hue amount

    :return: altered image
    """

    return tf.image.adjust_hue(image, delta)
