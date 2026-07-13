#!/usr/bin/env python3
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Calculate padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2

    elif padding == 'valid':
        ph = 0
        pw = 0

    elif isinstance(padding, tuple):
        ph, pw = padding

    else:
        raise ValueError("Invalid padding")

    # Add padding to images
    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )

    # Output dimensions
    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w))

    # Only two loops allowed
    for i in range(output_h):
        for j in range(output_w):

            # Extract the current window
            window = padded_images[
                :,
                i * sh:i * sh + kh,
                j * sw:j * sw + kw
            ]

            # Apply kernel
            output[:, i, j] = np.sum(
                window * kernel,
                axis=(1, 2)
            )

    return output
