#!/usr/bin/env python3
"""Module for performing a valid convolution on grayscale images."""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Perform a valid convolution on grayscale images.

    A valid convolution applies the kernel only where it completely
    overlaps the image. No padding is added to the images.

    Args:
        images (numpy.ndarray): Array of shape (m, h, w) containing
            multiple grayscale images.
            - m: number of images
            - h: image height
            - w: image width
        kernel (numpy.ndarray): Array of shape (kh, kw) containing
            the convolution kernel.
            - kh: kernel height
            - kw: kernel width

    Returns:
        numpy.ndarray: Array containing the convolved images with shape
        (m, h - kh + 1, w - kw + 1).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_height = h - kh + 1
    output_width = w - kw + 1

    output = np.zeros((m, output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
