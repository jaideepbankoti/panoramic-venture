"""
    References:
        https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
        https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function
        https://en.wikipedia.org/wiki/Bilinear_interpolation#Application_in_image_processing
"""

import numpy as np
import cv2

def perspective_warp(right_img, H, dim):
    """Works similar to cv2.warpPerspective method
    :param right_img: the right image
    :param H: homography matrix
    :param dim: the resulting dimension for composting
    :return: the warped image
    """
    col, row = dim
    Y, X = np.indices((row, col), dtype=np.float32)
    H = np.linalg.inv(H)        # computed the inverse homography, as we are looking for poistion of points in second image in first
    vec_indices = np.array([X.ravel(), Y.ravel(), np.ones_like(X).ravel()])     # creating a vector of homogeneous indices
    warped_indices = np.dot(H, vec_indices)     # calculating the indices after homography
    warpx, warpy = warped_indices[:-1] / warped_indices[-1]     # dividing by the scale
    warpx = warpx.reshape(row, col).astype(np.float32)          # reshaping the coordinates
    warpy = warpy.reshape(row, col).astype(np.float32)

    warp_img = cv2.remap(right_img, warpx, warpy, cv2.INTER_AREA)
    return warp_img