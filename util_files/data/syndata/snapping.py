from copy import deepcopy
from warnings import warn

from .types import ET_A, ET_B


def c2t(c): return c.real, c.imag


def t2c(t): return complex(*t)


def snap_beam(snap_what, snap_to, endpoint_type=ET_A):
    # TODO @artonson also return resulting junction types
    snapped_first = deepcopy(snap_what)
    first_svg = [line.to_svg() for line in snapped_first]
    second_svg = [line.to_svg() for line in snap_to]

    for line_prim, line_svg in zip(snapped_first, first_svg):
        for attempt in range(4):
            if   ET_A == endpoint_type: line_svg.start = line_svg.point(1 - 2 ** attempt)
            elif ET_B == endpoint_type: line_svg.end   = line_svg.point(2 ** attempt)
            intersection = line_svg.intersect(second_svg[-1])
            if intersection:
                t1, t2 = intersection[0]
                new_endpoint = c2t(line_svg.point(t1))
                if   ET_A == endpoint_type: line_prim.point1 = new_endpoint
                elif ET_B == endpoint_type: line_prim.point2 = new_endpoint
                break
        if not intersection:
            warn('failed to snap {} to {}'.format(line_svg, second_svg[-1]))

    return snapped_first


def snap_outer(snap_first, snap_second, endpoint_first=ET_A, endpoint_second=ET_A):
    # TODO @artonson also return resulting junction types
    snapped_first = deepcopy(snap_first)
    snapped_second = deepcopy(snap_second)
    first_svg = [line.to_svg() for line in snapped_first]
    second_svg = [line.to_svg() for line in snapped_second]

    for line_prim_first, line_svg_first, line_prim_second, line_svg_second in \
            zip(snapped_first, first_svg, snapped_second, second_svg):
        for attempt in range(4):
            if   ET_A == endpoint_first: line_svg_first.start = line_svg_first.point(1 - 2 ** attempt)
            elif ET_B == endpoint_first: line_svg_first.end   = line_svg_first.point(2 ** attempt)

            if   ET_A == endpoint_second: line_svg_second.start = line_svg_second.point(1 - 2 ** attempt)
            elif ET_B == endpoint_second: line_svg_second.end   = line_svg_second.point(2 ** attempt)

            intersection = line_svg_first.intersect(line_svg_second)
            if intersection:
                t1, t2 = intersection[0]
                new_endpoint = c2t(line_svg_first.point(t1))
                if   ET_A == endpoint_first: line_prim_first.point1 = new_endpoint
                elif ET_B == endpoint_first: line_prim_first.point2 = new_endpoint

                if   ET_A == endpoint_second: line_prim_second.point1 = new_endpoint
                elif ET_B == endpoint_second: line_prim_second.point2 = new_endpoint
                break

        if not intersection:
            warn('failed to snap {} to {}'.format(line_svg_first, line_svg_second))

    # snap the rest as beam
    first_len, second_len = len(snapped_first), len(snapped_second)
    if first_len > second_len:
        snapped_first[second_len:] = snap_beam(
            snapped_first[second_len:], snapped_second[-1:], endpoint_type=endpoint_first)
    elif first_len < second_len:
        snapped_second[first_len:] = snap_beam(
            snapped_second[first_len:], snapped_first[-1:], endpoint_type=endpoint_second)

    return snapped_first, snapped_second
