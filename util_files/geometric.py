import numpy as np

def flip_y(p): return (p[0], -p[1])
def flip_bb(p): return (p[0], -p[1], p[2], -p[3])


def liang_barsky_screen(point0, point1, bbox):
    point0_flip, point1_flip = flip_y(point0), flip_y(point1)
    bbox_flip = flip_bb(bbox)

    point0_clip, point1_clip, is_draw = liang_barsky_clipping(point0_flip, point1_flip, bbox_flip)
    if is_draw:
        point0_clip, point1_clip = flip_y(point0_clip), flip_y(point1_clip)
    return point0_clip, point1_clip, is_draw



def liang_barsky_clipping(point0, point1, bbox):
    # Based on
    # Liang-Barsky function by Daniel White @ http://www.skytopia.com/project/articles/compsci/clipping.html

    t0, t1 = 0., 1.
    point0, point1 = np.array(point0), np.array(point1)
    (x0, y0), (x1, y1) = point0, point1
    left, top, right, bottom = bbox
    xdelta, ydelta = point1 - point0
    for edge in range(4):           # Traverse through left, right, bottom, top edges.
        if edge == 0:
            p, q = -xdelta, -(left - x0)
        elif edge == 1:
            p, q = xdelta, (right - x0)
        elif edge == 2:
            p, q = -ydelta, -(bottom - y0)
        else:
            p, q = ydelta, (top - y0)

        if p == 0 and q < 0:
            return None, None, False            # Don't draw line at all. (parallel line outside)
        r = q / p

        if p < 0:
            if r > t1:
                return None, None, False        # Don't draw line at all
            elif r > t0:
                t0 = r                          # Line is clipped!
        elif p > 0:
            if r < t0:
                return None, None, False        # Don't draw line at all
            elif r < t1:
                t1 = r                          # Line is clipped!

    x0clip, y0clip = x0 + t0 * xdelta, y0 + t0 * ydelta
    x1clip, y1clip = x0 + t1 * xdelta, y0 + t1 * ydelta
    return (x0clip, y0clip), (x1clip, y1clip), True


def rotation_matrix_2d(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s), (s, c)))


def direction_from_angle(angle):
    return np.array([
        np.cos(angle), np.sin(angle)
    ])
