from itertools import combinations

import numpy as np
import scipy.ndimage as ndi

from .kanungo_degrade import kanungo_degrade_wrapper
import util_files.data.transforms.ocrodeg_degrade as ocrodeg


def page_rotate(image):
    """Rotate the image randomly by 0, 90, 180, or 270 degrees."""
    angle = np.random.choice([0, 90, 180, 270])
    return ndi.rotate(image, angle)

def _random_geometric_transform_wrapper(image):
    transform = ocrodeg.random_transform(
        translation=(-0.005, 0.005),
        rotation=(-.2, .2),
        scale=(-0.01, 0.01),
        aniso=(-0.01, 0.01))
    return ocrodeg.transform_image(image, **transform)

def _distort_wrapper(image):
    """Distorts image with bounded noise."""
    sigma = np.random.uniform(1., 2.)
    maxdelta = np.random.uniform(1., 2.)
    noise = ocrodeg.bounded_gaussian_noise(image.shape, sigma, maxdelta)
    return ocrodeg.distort_with_noise(image, noise)

def _ruled_distort_wrapper(image):
    """Distorts image with 1d bounded noise."""
    magnitude = np.random.uniform(50., 50.)
    noise = ocrodeg.noise_distort1d(image.shape, magnitude=magnitude)
    return ocrodeg.distort_with_noise(image, noise)

def _gaussian_blur_wrapper(image):
    """Blurs the image."""
    s = np.random.uniform(0, 3)
    return ndi.gaussian_filter(image, s)

def _thresholding_wrapper(image):
    """Thresholds the image."""
    t = np.random.uniform(.5, 1.)
    return 1. * (image > t)

def _binary_blur_wrapper(image):
    """Glues stuff together"""
    blur = np.random.uniform(0, 3)
    return ocrodeg.binary_blur(image, blur)

def _noisy_binary_blur_wrapper(image):
    """Glues stuff together, adding shot noise."""
    blur = np.random.uniform(0, 1.5)
    sigma = np.random.uniform(0, .3)
    return ocrodeg.binary_blur(image, blur, noise=sigma)

def _random_blotches_wrapper(image):
    """Places random blobs in foreground and background."""
    fgblobs = np.random.uniform(1, 5) * 1e-4
    fgscale = int(np.random.uniform(5, 20))
    bgblobs = np.random.uniform(.5, 3) * 1e-4
    bgscale = int(np.random.uniform(5, 20))
    return ocrodeg.random_blotches(image, fgblobs, bgblobs, fgscale=fgscale, bgscale=bgscale)

def _no_degradation(image): return image


degradations_by_name = {
    'kanungo': kanungo_degrade_wrapper,
    # 'page_rotation': page_rotate,
    'random_geometric': _random_geometric_transform_wrapper,
    # 'random_geometric_rotate': random_geometric_rotate,
    # 'random_geometric_aniso': random_geometric_aniso,
    # 'random_geometric_scale': random_geometric_scale,
    'distort': _distort_wrapper,
    # 'ruled_distort': _ruled_distort_wrapper,
    'gaussian_blur': _gaussian_blur_wrapper,
    'thresholding': _thresholding_wrapper,
    'binary_blur': _binary_blur_wrapper,
    'noisy_binary_blur': _noisy_binary_blur_wrapper,
    'random_blotches': _random_blotches_wrapper,
    'nothing': _no_degradation,
}

all_degradations = list(degradations_by_name.keys())


class DegradationGenerator:
    def __init__(self, degradations_list, max_num_degradations=2):
        self.degradations_list = degradations_list
        self.max_num_degradations = max_num_degradations
        self.degradations = []
        for i in range(max_num_degradations):
            self.degradations.extend(
                list(combinations(self.degradations_list, i + 1))
            )

    def do_degrade(self, image):
        """Degrade image using a random combination of degradations.

        :param image: input image
        :type image: numpy.ndarray of shape [h, w] with values in range [0, 255)
        :returns: degraded image
        :rtype: numpy.ndarray of shape [h, w] with values in range [0, 255)
        """
        degradations = self.degradations[np.random.choice(len(self.degradations))]
        result = np.copy(image)
        for deg_name in degradations:
            deg_applier = degradations_by_name[deg_name]
            result = deg_applier(result)
        return result

    def __call__(self, image):
        return self.do_degrade(image)
