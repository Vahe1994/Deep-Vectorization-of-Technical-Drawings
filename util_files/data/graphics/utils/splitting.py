import numpy as np

from util_files.data.graphics.primitives import Line, CBezier, QBezier
from . common import bbox


def split_by_line(segs, line):
    '''Returns subsegments_on_the_left, subsegments_on_the_right'''
    l0, l1 = np.asarray(line)
    assert np.any(l0 != l1)

    l = l1 - l0
    l = l / np.linalg.norm(l)
    n = np.roll(l, 1, axis=-1);  n[0] *= -1 # normal is 90 degrees counterclockwise, i.e it is directed to the left
    dist = lambda p: (p.real - l0[0]) * n[0] + (p.imag - l0[1]) * n[1]

    left_segs = []
    right_segs = []
    for seg in segs:
        # the whole segment lies on the line
        bpoints = np.array([[p.real, p.imag] for p in seg.bpoints()])
        bpoint_dists = (bpoints - l0).dot(n)
        if np.all(bpoint_dists == 0):
            # put the segment in the uppermost or leftmost half-space
            if n[1] < 0:
                left_segs.append(seg)
                continue
            elif n[1] > 0:
                right_segs.append(seg)
                continue
            else:
                if n[0] < 0:
                    left_segs.append(seg)
                    continue
                else:
                    right_segs.append(seg)
                    continue

        for subseg in seg.split_with_infline(l0, n):
            if isinstance(subseg, Line):
                if any(dist(p) > 0 for p in subseg.bpoints()):
                    left_segs.append(subseg)
                    continue
                else:
                    right_segs.append(subseg)
                    continue
            elif isinstance(subseg, (CBezier, QBezier)): # doesn't work for quartics and higher order
                if any(dist(p) > 0 for p in [subseg.start, subseg.end, subseg.point(.5)]):
                    left_segs.append(subseg)
                    continue
                else:
                    right_segs.append(subseg)
                    continue
            else:
                raise NotImplementedError('Don\'t know how to split {}'.format(subseg.__class__))

    return left_segs, right_segs


def split_to_patches(segments, origin, patch_size, patches_n):
    '''Returns i,j,v analogous to "coo" sparse format'''
    patch_w, patch_h = patch_size
    patches_row_n, patches_col_n = patches_n

    minx, maxx, miny, maxy = bbox(segments)
    mini = max(int(np.floor((miny - origin[1]) / patch_h)), 0)
    maxi = min(int(np.ceil((maxy - origin[1]) / patch_h)), patches_row_n)
    minj = max(int(np.floor((minx - origin[0]) / patch_w)), 0)
    maxj = min(int(np.ceil((maxx - origin[0]) / patch_w)), patches_col_n)

    lower_boundary = [[0, origin[1]], [1, origin[1]]]
    segments, _ = split_by_line(segments, lower_boundary)

    left_boundary = [[origin[0], 0], [origin[0], -1]]
    segments, _ = split_by_line(segments, left_boundary)

    iS = []
    jS = []
    patches = []

    upper_segments = segments
    for patch_i in range(mini, maxi):
        upper_y = origin[1] + patch_h * (patch_i + 1)
        upper_boundary = [[0, upper_y], [-1, upper_y]]
        lower_segments, upper_segments = split_by_line(upper_segments, upper_boundary)

        right_segments = lower_segments
        for patch_j in range(minj, maxj):
            right_x = origin[0] + patch_w * (patch_j + 1)
            right_boundary = [[right_x, 0], [right_x, 1]]
            left_segments, right_segments = split_by_line(right_segments, right_boundary)
            if len(left_segments) != 0:
                iS.append(patch_i)
                jS.append(patch_j)
                patches.append(left_segments)

    return iS, jS, patches


def crop_to_bbox(segments, bbox):
    min_x, max_x, min_y, max_y = bbox
    _, segments = split_by_line(segments, [[min_x, 0], [min_x, 1]])
    _, segments = split_by_line(segments, [[0, max_y], [1, max_y]])
    _, segments = split_by_line(segments, [[max_x, 1], [max_x, 0]])
    _, segments = split_by_line(segments, [[1, min_y], [0, min_y]])
    return segments
