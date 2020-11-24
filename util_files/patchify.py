# Taken from https://github.com/dovahcrow/patchify.py/blob/master/patchify.py

from itertools import product
from typing import Tuple

import numpy as np
from skimage.util import view_as_windows


def patchify(patches: np.ndarray, patch_size: Tuple[int, int], step: int = 1):

    return view_as_windows(patches, patch_size, step)


def unpatchify(patches: np.ndarray, imsize: Tuple[int, int]):
    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape

    # Calculate the overlap size in each axis
    o_w = (n_w * p_w - i_w) / (n_w - 1)
    o_h = (n_h * p_h - i_h) / (n_h - 1)

    # The overlap should be integer, otherwise the patches are unable to reconstruct into a image with given shape
    assert int(o_w) == o_w
    assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i, j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    return image / divisor


def split_to_patches(rgb, patch_size, overlap=0):
    """Separates the input into square patches of specified size.

    :param rgb: input RGB image
    :type rgb: numpy.ndarray
    :param patch_size: size of patches in pixels (assuming
                        square patches)
    :type patch_size: int
    :param overlap: amount in pixels of how much the patches
                    can overlap with each other (useful for merging)
    :type overlap: int

    :returns patches, patches_offsets
    :rtype Tuple[numpy.ndarray, numpy.ndarray]
    """
    rgb = rgb.transpose(1, 2, 0)
    rgb_t = np.ones((rgb.shape[0] + 33, rgb.shape[1] + 33, rgb.shape[2])) * 255.
    rgb_t[:rgb.shape[0], :rgb.shape[1], :] = rgb
    rgb = rgb_t

    height, width, channels = rgb.shape

    assert patch_size > 0 and 0 <= overlap < patch_size
    patches = patchify(rgb,
                       patch_size=(patch_size, patch_size, channels),
                       step=patch_size - overlap)
    patches = patches.reshape((-1, patch_size, patch_size, channels))
    height_offsets = np.arange(0, height - patch_size, step=patch_size - overlap)
    width_offsets = np.arange(0, width - patch_size, step=patch_size - overlap)
    patches_offsets = np.array(list(
        product(height_offsets, width_offsets)
    ))
    return patches, patches_offsets, rgb