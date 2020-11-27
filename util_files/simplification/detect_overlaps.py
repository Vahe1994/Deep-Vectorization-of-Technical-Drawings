from itertools import groupby

import numpy as np
# import svgpathtools
#
# from vectran.data.graphics_primitives import PT_LINE, PT_QBEZIER


def has_overlaps(primitives, render, extra_line_width=.5, max_relative_overlap=.5):  # max_relative_overlap=.9, max_absolute_overlap=10):
    primitives = [(prim_type, prim) for prim_type, prim_set in primitives.items() for prim in prim_set]
    primitives = np.array(primitives)
    get_type = lambda x: x[0]
    get_prim = lambda x: x[1]

    for i in range(len(primitives)):
        prim_type, primitive = primitives[i]
        primitive = np.array(primitive)
        other_primitives = primitives[np.arange(len(primitives)) != i]
        other_primitives = {prim_type: np.asarray(list(map(get_prim, primitives_subset))) for
                            prim_type, primitives_subset in groupby(other_primitives, get_type)}
        primitive[-1] += extra_line_width
        overlap = get_overlap_ratio({prim_type: [primitive]}, other_primitives, render)
        if overlap > max_relative_overlap:
            return True
        # if (overlap > max_relative_overlap) or (overlap * prim_length(prim_type, primitive) > max_absolute_overlap):
        #     return True
    return False


def get_overlap_ratio(primitive, other_primitives, render):
    # 1. Render the primitive and the other primitives
    primitive_rendering = np.asarray(render(primitive))
    others_rendering = np.asarray(render(other_primitives))

    # 2. Calculate the area covered by this primitive
    primitive_rendering = (255 - primitive_rendering).astype(np.float32)
    area = primitive_rendering.sum() / 255
    #    and the area of overlap
    covered_by_others = others_rendering < 255
    overlap_area = primitive_rendering[covered_by_others].sum() / 255

    return overlap_area / area


# def prim_length(prim_type, prim):
#     return _prim_length[prim_type](prim)
#
#
# def line_length(line):
#     start = line[0] + line[1] * 1j
#     end = line[2] + line[3] * 1j
#     return svgpathtools.Line(start, end).length()
#
#
# def qbezier_length(qbezier):
#     p1 = qbezier[0] + qbezier[1] * 1j
#     p2 = qbezier[2] + qbezier[3] * 1j
#     p3 = qbezier[4] + qbezier[5] * 1j
#     return svgpathtools.QuadraticBezier(p1, p2, p3).length()
#
#
# _prim_length = {PT_LINE: line_length, PT_QBEZIER: qbezier_length}
