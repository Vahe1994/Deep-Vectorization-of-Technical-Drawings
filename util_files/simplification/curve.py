import numpy as np
import svgpathtools

from .utils import sqlen


def bezier_steps(bezier, same_line_threshold):
    # source: https://github.com/servo/cairo/blob/master/src/cairo-mesh-pattern-rasterizer.c
    if isinstance(bezier, svgpathtools.CubicBezier):
        p0, p1, p2, p3 = (np.array((p.real, p.imag)) / same_line_threshold for p in bezier.bpoints())
    elif isinstance(bezier, svgpathtools.QuadraticBezier):
        q0, q1, q2 = (np.array((p.real, p.imag)) / same_line_threshold for p in bezier.bpoints())
        p0 = q0
        p1 = (q0 + q1 * 2) / 3
        p2 = (q2 + q1 * 2) / 3
        p3 = q2
    sqsteps = 18 * max(sqlen(p0 - p1), sqlen(p2 - p3), sqlen(p0 - p2) * .25, sqlen(p1 - p3) * .25)
    return int(np.ceil(np.sqrt(sqsteps)))


def find_longest_flat(poly_x, poly_y, ts, threshold):
    '''Returns `line, N` such that each of the first N points are no further than threshold from the line
       and `line = points[0], points[N-1]`'''
    assert len(ts) > 1

    t0 = ts[0]
    p0 = poly_x(t0), poly_y(t0)

    for N in range(len(ts), 1, -1):
        t1 = ts[N - 1]
        line = p0, (poly_x(t1), poly_y(t1))
        if polycurve_is_in_line(poly_x, poly_y, line, threshold, tmin=t0, tmax=t1):
            return line, N

    return None, 0


def polycurve_is_in_line(poly_x, poly_y, line, max_deviation, tmin=0, tmax=1):
    line = np.asarray(line)
    if np.all(line[0] == line[1]):  # degenerate line
        return polycurve_is_in_point(poly_x, poly_y, line[0], max_deviation, tmin=tmin, tmax=tmax)

    l = line[1] - line[0]
    length = np.linalg.norm(l)
    l /= length

    l0x, l0y = line[0]
    poly_proj = (poly_x - l0x) * l[0] + (poly_y - l0y) * l[1]
    poly_dist = (poly_y - l0y) * l[0] - (poly_x - l0x) * l[1]

    additional_roots = np.array([tmin, tmax])

    # check the interior of the line
    roots = poly_dist.deriv().roots
    roots = roots[np.isreal(roots)].real
    roots = np.concatenate((roots[(roots >= tmin) & (roots <= tmax)], additional_roots))
    if np.any(np.abs(poly_dist(roots)) > max_deviation):
        return False

    max_deviation_sq = max_deviation ** 2

    # check the left end of the line
    poly_sqdist_l0 = (poly_x - l0x) ** 2 + (poly_y - l0y) ** 2
    roots = poly_sqdist_l0.deriv().roots
    roots = roots[np.isreal(roots)].real
    roots = np.concatenate((roots[(roots >= tmin) & (roots <= tmax)], additional_roots))
    roots = roots[poly_proj(roots) < 0]
    if np.any(poly_sqdist_l0(roots) > max_deviation_sq):
        return False

    # check the right end of the line
    l1x, l1y = line[1]
    poly_sqdist_l1 = (poly_x - l1x) ** 2 + (poly_y - l1y) ** 2
    roots = poly_sqdist_l1.deriv().roots
    roots = roots[np.isreal(roots)].real
    roots = np.concatenate((roots[(roots >= tmin) & (roots <= tmax)], additional_roots))
    roots = roots[poly_proj(roots) > length]
    return np.all(poly_sqdist_l1(roots) <= max_deviation_sq)


def polycurve_is_in_point(poly_x, poly_y, point, max_deviation, tmin=0, tmax=1):
    poly_sqdist = (poly_x - point[0]) ** 2 + (poly_y - point[1]) ** 2
    max_deviation_sq = max_deviation ** 2
    if (poly_sqdist(tmin) > max_deviation_sq) or (poly_sqdist(tmax) > max_deviation_sq):
        return False

    roots = poly_sqdist.deriv().roots
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= tmin) & (roots <= tmax)]
    return np.all(poly_sqdist(roots) <= max_deviation_sq)
