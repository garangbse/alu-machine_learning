#!/usr/bin/env python3
"""Module for performing same convolution on grayscale images."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Perform a same convolution on grayscale images.

    The images are padded with zeros so that the output has the same
    height and width as the input images.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w) containing
            grayscale images.
        kernel (numpy.ndarray): Array of shape (kh, kw) containing the
            convolution kernel.

    Returns:
        numpy.ndarray: Array containing the convolved images with shape
        (m, h, w).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    if kh % 2 == 0:
        pad_top = kh // 2
        pad_bottom = (kh // 2) - 1
    else:
        pad_top = kh // 2
        pad_bottom = kh // 2

    if kw % 2 == 0:
        pad_left = kw // 2
        pad_right = (kw // 2) - 1
    else:
        pad_left = kw // 2
        pad_right = kw // 2

    padded = np.pad(
        images,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant'
    )

    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
