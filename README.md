# 0. Valid Convolution

## Description

This project implements a function that performs a **valid convolution** on multiple grayscale images using **NumPy**.

A **valid convolution** applies a kernel (filter) to an image **without padding**. The kernel is only applied where it fits entirely inside the image, resulting in an output image that is smaller than the input.

The implementation supports processing multiple images simultaneously through NumPy vectorization.

## Function Prototype

```python
def convolve_grayscale_valid(images, kernel):
```

## Parameters

### `images`

A NumPy array of shape:

```text
(m, h, w)
```

where:

* `m` is the number of grayscale images.
* `h` is the height of each image.
* `w` is the width of each image.

### `kernel`

A NumPy array of shape:

```text
(kh, kw)
```

where:

* `kh` is the kernel height.
* `kw` is the kernel width.

## Return Value

Returns a NumPy array containing the convolved images with shape:

```text
(m, h - kh + 1, w - kw + 1)
```

## How It Works

1. Determine the dimensions of the images and kernel.
2. Compute the dimensions of the output.
3. Slide the kernel across every valid position in each image.
4. Multiply the image region element-wise with the kernel.
5. Sum the resulting values.
6. Store the result in the corresponding output position.

## Example

Input image:

```text
1 2 3
4 5 6
7 8 9
```

Kernel:

```text
1 0
0 1
```

Output:

```text
6  8
12 14
```

## Requirements

* Python 3
* NumPy
* No use of `np.convolve`
* Only `import numpy as np` is allowed
* Code follows Pycodestyle
* All modules and functions are documented

## Author

Garang Buke
