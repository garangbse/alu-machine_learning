#!/usr/bin/env python3
"""Module for performing convolution on grayscale images with padding."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Perform a convolution on grayscale images using custom padding.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w) containing
            multiple grayscale images.
        kernel (numpy.ndarray): Array of shape (kh, kw) containing
            the convolution kernel.
        padding (tuple): Tuple (ph, pw) where:
            - ph is the padding for the height.
            - pw is the padding for the width.

    Returns:
        numpy.ndarray: The convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    output_h = h + (2 * ph) - kh + 1
    output_w = w + (2 * pw) - kw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = padded[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
