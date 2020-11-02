import numpy as np
from skimage.color import rgb2gray


def rgb_to_gray(rgb):
    """Convert [h, w, 3] 8-bit RGB image to [h, w] 8-bit grayscale image."""
    gray_float = rgb2gray(img_8bit_to_float(rgb))
    return gray_float_to_8bit(gray_float)


def ensure_gray_8bit(image):
    """Check if the image has 3 channels, convert to gray scale if so."""
    if image.ndim == 3:
        image = rgb_to_gray(image)
    assert image.ndim == 2
    return image


def img_8bit_to_float(img):
    """Convert [h, w, ?] 8-bit grayscale image to [h, w, ?] [0, 1] float grayscale image.
    Does not change the image if it already is floating point-based."""
    return img.astype(np.float32) / np.amax(img)


def gray_float_to_8bit(gray):
    """Convert [h, w, ?] [0, 1] float grayscale image to [h, w, ?] 8-bit grayscale image."""
    return (gray * 255).astype(np.uint8)


def img_8bit_to_binary_maxwhite(gray):
    """Binarize the image, converting all grayscale values to white
    (thinning the lines, maximizing whitespace)."""
    return np.where(gray > 0, 1, 0).astype(np.uint8)


def img_8bit_to_binary_maxink(gray):
    """Binarize the image, converting all grayscale values to black
    (thickening the lines, maximizing ink)."""
    return np.where(gray == 255, 1, 0).astype(np.uint8)


def img_8bit_to_binary_median(gray):
    """Binarize the image, converting all values below 128 to black."""
    return np.where(gray > 127, 1, 0).astype(np.uint8)