"""
This module contains common visualization functions
used to report results of the models.
"""

from functools import partial
import numpy as np


def horiz_merge(left, right):
    """
    merges two images, left and right horizontally to obtain
    a bigger image containing both.

    Parameters
    ---------
    left: 2D or 3D numpy array
        left image.
        2D for grayscale.
        3D for color.
    right : numpy array array
        right image.
        2D for grayscale
        3D for color.

    Returns
    -------

    numpy array (2D or 3D depending on left and right)
    """
    assert left.shape[0] == right.shape[0]
    assert left.shape[2:] == right.shape[2:]
    shape = (left.shape[0], left.shape[1] + right.shape[1],) + left.shape[2:]
    im_merge = np.zeros(shape)
    im_merge[:, 0:left.shape[1]] = left
    im_merge[:, left.shape[1]:] = right
    return im_merge

def vert_merge(top, bottom):
    """
    merges two images, top and bottom vertically to obtain
    a bigger image containing both.

    Parameters
    ---------
    top: 2D or 3D numpy array
        top image.
        2D for grayscale.
        3D for color.
    bottom : numpy array array
        bottom image.
        2D for grayscale
        3D for color.

    Returns
    -------

    numpy array (2D or 3D depending on left and right)
    """
    im = horiz_merge(top, bottom)
    if len(im.shape) == 2:
        im = im.transpose((1, 0))
    elif len(im.shape) == 3:
        im = im.transpose((1, 0, 2))
    return im


def grid_of_images(M, border=0, bordercolor=[0.0, 0.0, 0.0], shape=None, normalize=False):
    """
    Draw a grid of images from M
    The order in the grid which corresponds to the order in M
    is starting from top to bottom then left to right.

    Parameters
    ----------

    M : numpy array
        if 3D, convert it to 4D, the shape will be interpreted as (nb_images, h, w) and converted to (nb_images, 1, h, w).
        if 4D, consider it as colored or grayscale
            - if the shape is (nb_images, nb_colors, h, w), it is converted to (nb_images, h, w, nb_colors)
            - otherwise, if it already (nb_images, h, w, nb_colors), use it as it is.
            - nb_colors can be 1 (grayscale) or 3 (colors).
    border: int
        thickness of border(default=0)
    shape: tuple (nb_cols, nb_rows)
        shape of the grid
        by default make a square shape
        (in that case, it is possible that not all images from M will be part of the grid).
    normalize: bool(default=False)
        whether to normalize the pixel values of each image independently
        by min and max. if False, clip the values of pixels to 0 and 1
        without normalizing.

    Returns
    -------

    3D numpy array of shape (h, w, 3)
    (with a color channel regardless of whether the original images were grayscale or colored)
    """
    if len(M.shape) == 3:
        M = M[:, :, :, np.newaxis]
    if M.shape[-1] not in (1, 3):
        M = M.transpose((0, 2, 3, 1))
    if M.shape[-1] == 1:
        M = np.ones((1, 1, 1, 3)) * M
    bordercolor = np.array(bordercolor)[None, None, :]
    numimages = len(M)
    M = M.copy()

    if normalize:
        for i in range(M.shape[0]):
            M[i] -= M[i].flatten().min()
            M[i] /= M[i].flatten().max()
    else:
        M = np.clip(M, 0, 1)
    height, width, color = M[0].shape
    assert color == 3, 'Nb of color channels are {}'.format(color)
    if shape is None:
        n0 = np.int(np.ceil(np.sqrt(numimages)))
        n1 = np.int(np.ceil(np.sqrt(numimages)))
    else:
        n0 = shape[0]
        n1 = shape[1]

    im = np.array(bordercolor) * np.ones(
        ((height + border) * n1 + border, (width + border) * n0 + border, 1), dtype='<f8')
    # shape = (n0, n1)
    # j corresponds to rows in the grid, n1 should correspond to nb of rows
    # i corresponds to columns in the grid, n0 should correspond to nb of cols
    # M should be such that the first n1 examples correspond to row 1, 
    # next n1 examples correspond to row 2, etc. that is, M first axis
    # can be reshaped to (n1, n0)
    for i in range(n0):
        for j in range(n1):
            if i * n1 + j < numimages:
                im[j * (height + border) + border:(j + 1) * (height + border) + border,
                   i * (width + border) + border:(i + 1) * (width + border) + border, :] = np.concatenate((
                       np.concatenate((M[i * n1 + j, :, :, :],
                                       bordercolor * np.ones((height, border, 3), dtype=float)), 1),
                       bordercolor * np.ones((border, width + border, 3), dtype=float)
                   ), 0)
    return im

grid_of_images_default = partial(grid_of_images, border=1, bordercolor=(0.3, 0, 0))


def reshape_to_images(x, input_shape=None):
    """
    a function that takes a numpy array and try to
    reshape it to an array of images that would
    be compatible with the function grid_of_images.
    Two cases are considered.

    if x is a 2D numpy array, it uses input_shape:
        - x can either be (nb_examples, nb_features) or (nb_features, nb_examples)
        - nb_features should be prod(input_shape)
        - the nb_features dim is then expanded to have :
            (nb_examples, h, w, nb_channels), sorted input_shape shoud
            be  (h, w, nb_channels).

    if x is a 4D numpy array:
        - if the first tensor dim is 1 or 3 like e.g. (1, a, b, c), then assume it is 
          color channel and transform to (a, 1, b, c)
        - if the second tensor dim is 1 or 3, leave x it as it is
        - if the third tensor dim is 1 or 3, like e.g. (a, b, 1, c), then assume it is
          color channel and transform to (c, 1, a, b)
        - if the fourth tensor dim is 1 or 3, like e.g. (a, b, c, 1), then assume it is
          color channel and transform to (c, 1, a, b)
    Parameters
    ----------

    x : numpy array
        input to be reshape
    input_shape : tuple needed only when x is 2D numpy array
    """
    if len(x.shape) == 2:
        assert input_shape is not None
        if x.shape[0] == np.prod(input_shape):
            x = x.T
            x = x.reshape((x.shape[0],) + input_shape)
            x = x.transpose((0, 2, 3, 1))
            return x
        elif x.shape[1] == np.prod(input_shape):
            x = x.reshape((x.shape[0],) + input_shape)
            x = x.transpose((0, 2, 3, 1))
            return x
        else:
            raise ValueError('Cant recognize this shape : {}'.format(x.shape))
    elif len(x.shape) == 4:
        if x.shape[0] in (1, 3):
            x = x.transpose((1, 0, 2, 3))
            return x
        elif x.shape[1] in (1, 3):
            return x
        elif x.shape[2] in (1, 3):
             x = x.transpose((3, 2, 0, 1))
             return x
        elif x.shape[3] in (1, 3):
            x = x.transpose((2, 3, 0, 1))
            return x
        else:
            raise ValueError('Cant recognize a shape of size : {}'.format(len(x.shape)))
    else:
        raise ValueError('Cant recognize a shape of size : {}'.format(len(x.shape)))
