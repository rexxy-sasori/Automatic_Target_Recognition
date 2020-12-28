import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def elastic_transform(image, alpha=10, sigma=3, random_state=None):
    """
        Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    row, col = shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    pts1 = np.zeros((row * col, 3))

    pts1[:, 0] = indices[0].reshape((indices[0].shape[0] * indices[0].shape[1]))
    pts1[:, 1] = indices[1].reshape((indices[1].shape[0] * indices[1].shape[1]))
    pts1[:, 2] = image.reshape((row * col))

    spline = map_coordinates(image, indices, order=3).reshape(shape)
    return spline


def img_read(file_name, crop_size=0):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ROW, COL = img.shape

    row_start = crop_size
    row_end = ROW - crop_size
    col_start = crop_size
    col_end = COL - crop_size
    img = img[row_start:row_end, col_start:col_end]
    return img
