import numpy as np


def qbezier_to_cbezier(qbezier):
    p0 = np.asarray(qbezier)[..., :2]
    qp1 = np.asarray(qbezier)[..., 2:4]
    qp2 = np.asarray(qbezier)[..., 4:6]
    rest = np.asarray(qbezier)[..., 6:]
    p1 = (qp1 * 2 + p0) / 3
    p2 = (qp1 * 2 + qp2) / 3
    return np.concatenate([p0, p1, p2, qp2, rest], axis=-1)
