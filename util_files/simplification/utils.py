import numpy as np


def pointsC_are_indistinguishable(p1, p2, distinguishability_threshold):
    return abs(p2 - p1) < distinguishability_threshold


def sqlen(l):
    return l.dot(l)


def unique_points(points, return_inverse=False):
    if return_inverse:
        unique_points, inverse_ids = np.unique(points.view(np.complex64), return_inverse=True)
        unique_points = unique_points.view(np.float32).reshape(-1, 2)
        return unique_points, inverse_ids
    else:
        return np.unique(points.view(np.complex64)).view(np.float32).reshape(-1, 2)
