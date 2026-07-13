#!/usr/bin/env python3
"""
Module for performing convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w)
        kernel: numpy.ndarray with shape (kh, kw)
        padding: tuple (ph, pw), 'same', or 'valid'
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray containing the convolved images.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'same':
        output_h = int(np.ceil(h / sh))
        output_w = int(np.ceil(w / sw))

        pad_h = max((output_h - 1) * sh + kh - h, 0)
        pad_w = max((output_w - 1) * sw + kw - w, 0)

        ph = pad_h // 2
        pw = pad_w // 2

    elif padding == 'valid':
        ph = 0
        pw = 0

    elif isinstance(padding, tuple):
        ph, pw = padding

    else:
        raise ValueError("Invalid padding")

    # Pad images with zeros
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Output dimensions
    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, output_h, output_w))

    # Only two loops allowed
    for i in range(output_h):
        for j in range(output_w):

            window = images_padded[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw
            ]

            output[:, i, j] = np.sum(
                window * kernel,
                axis=(1, 2)
            )

    return output
