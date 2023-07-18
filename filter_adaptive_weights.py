import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.ndimage import convolve


def Gaussian_weight(sigma=3.0, size=15):

    G_kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            G_kernel[i, j] = np.exp(-0.5 * (euclidean((i, j), (size // 2, size // 2)) ** 2) / (sigma ** 2))

    return G_kernel

def W_k(img, G_kernel):

    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    gradient_xy = gradient_x + gradient_y
    gradient_xy_abs = np.abs(gradient_x) + np.abs(gradient_y)
    w = np.abs(convolve(gradient_xy, G_kernel, mode='constant'))/convolve(gradient_xy_abs, G_kernel, mode='constant')

    return w