import numpy as np
import skimage.color
import skimage.data
import scipy.signal
import matplotlib.pyplot as plt



def gaussian_blur(img, kernel_size):

    filter = np.array([1, 2, 1]) / 4.
    kernel = filter
    while kernel_size > 3:
        kernel = np.convolve(kernel, filter)
        kernel_size -= 2
    kernel = np.outer(kernel, kernel)
    blur = scipy.signal.convolve2d(img, kernel, 'valid')

    return blur

# Test

img = skimage.color.rgb2gray(skimage.data.astronaut())
plt.imshow(gaussian_blur(img, 13), cmap='gray')
plt.show()
plt.imshow(img, cmap='gray')
plt.show()