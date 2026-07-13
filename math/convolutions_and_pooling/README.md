# Convolutions and Pooling

## Description

This project introduces the fundamental operations used in Convolutional Neural Networks (CNNs): **convolutions** and **pooling**.

The tasks involve implementing convolution and pooling algorithms from scratch using only NumPy. The goal is to understand how kernels extract features from images and how pooling reduces spatial dimensions while preserving important information.

## Learning Objectives

By the end of this project, you should understand:

* What a convolution is
* What a kernel (filter) is
* How convolution is performed on grayscale images
* The purpose of padding
* The difference between **valid** and **same** padding
* What a stride is
* What image channels are
* How max pooling works
* How average pooling works

## Project Tasks

* 0. Valid Convolution
* 1. Same Convolution
* 2. Convolution with Padding
* 3. Convolution with Stride
* 4. Convolution with Channels
* 5. Convolution over Multiple Kernels
* 6. Pooling

## Technologies

* Python 3
* NumPy

## Requirements

* Ubuntu 16.04 LTS
* Python 3.5
* NumPy 1.15
* Pycodestyle 2.5
* All files begin with:

```text
#!/usr/bin/env python3
```

* Only `import numpy as np` and any explicitly allowed imports may be used.
* `np.convolve` is not allowed.
* Every module and function must be fully documented.

## Repository Structure

```text
math/
└── convolutions_and_pooling/
    ├── 0-convolve_grayscale_valid.py
    ├── 1-convolve_grayscale_same.py
    ├── 2-convolve_grayscale_padding.py
    ├── 3-convolve_grayscale.py
    ├── 4-convolve_channels.py
    ├── 5-convolve.py
    ├── 6-pool.py
    └── README.md
```

## Author

Garang Buke
