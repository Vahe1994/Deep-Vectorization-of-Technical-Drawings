import numpy as np
from scipy.ndimage import distance_transform_cdt, binary_closing


def threshold_gray_float(gray, t):
    binary = np.copy(gray)
    binary[gray > t] = 1.
    binary[gray <= t] = 0.
    return binary


def kanungo_degrade_wrapper(gray):
    binary = threshold_gray_float(gray, 1e-3)
    if np.sum(binary) == np.prod(binary.shape):
        return gray
    return kanungo_degrade(binary)


def kanungo_degrade(image, eta=0.0, alpha=1.5, beta=1.5, alpha_0=1.0, beta_0=1.0, do_closing=True):
    """Degrade image using method from
    Kanungo, T., Haralick, R. M., Baird, H. S., Stuezle, W., & Madigan, D. (2000).
    A statistical, nonparametric methodology for document degradation model validation.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11), 1209-1223.

    :param image: [h, w] 8-bit grayscale image
    :param eta:
    :param alpha:
    :param beta:
    :param alpha_0:
    :param beta_0:
    :param do_closing: whether to perform morphological closing of the result
    :return:
    """

    # invert and
    image = np.amax(image) - image
    image = image / np.amax(image)

    # flip foreground pixels
    fg_dist = distance_transform_cdt(image, metric='taxicab')
    fg_prob = alpha_0 * np.exp(-alpha * (fg_dist ** 2)) + eta
    fg_prob[image == 0] = 0
    fg_flip = np.random.binomial(1, fg_prob)

    bg_dist = distance_transform_cdt(1 - image, metric='taxicab')
    bg_prob = beta_0 * np.exp(-beta * (bg_dist ** 2)) + eta
    bg_prob[image == 1] = 0
    bg_flip = np.random.binomial(1, bg_prob)

    bg_mask = np.ma.make_mask(bg_flip)
    fg_mask = np.ma.make_mask(fg_flip)
    result = np.copy(image)
    result[bg_mask] = 1 - result[bg_mask]
    result[fg_mask] = 1 - result[fg_mask]

    if do_closing:
        sel = np.array([
            [1, 1],
            [1, 1],
        ])
        result = binary_closing(result, sel)

    # result = 255 - result.astype(np.uint8) * 255
    result = 1. - result.astype('float32')

    return result
