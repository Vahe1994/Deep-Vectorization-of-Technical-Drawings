from abc import ABC, abstractmethod
from copy import copy, deepcopy

import numpy as np
import svgpathtools

from .utils.common import mirror_point_x
from util_files.simplification.utils import pointsC_are_indistinguishable, unique_points
from util_files.simplification.simplify import bezier as simplify_bezier

INTERNAL_PRECISION = 4


class Primitive(ABC):
    @classmethod
    def from_seg(cls, seg):
        return cls._seg2prim[seg.__class__](seg)

    def copy(self):
        return copy(self)

    def bbox(self):
        return self.segment.bbox()

    def bpoints(self):
        return self.segment.bpoints()

    def cropped(self, t0, t1):
        return self.__class__(self.segment.cropped(t0, t1))

    def mirror(self, x):
        self.segment = self.segment.__class__(*(mirror_point_x(p, x) for p in self.segment.bpoints()))

    def point(self, t):
        return self.segment.point(t)

    def __repr__(self):
        return self.segment.__repr__()

    def rotate(self, degs, origin):
        self.segment = self.segment.rotated(degs, complex(*origin))

    def scale(self, scale):
        self.segment = self.segment.scaled(scale)

    def translate(self, t):
        self.segment = self.segment.translated(complex(*t))

    def to_svgpathtools(self):
        return deepcopy(self.segment)

    def vahe_representation(self, width):
        points = self.segment.bpoints()
        reversed_points = reversed(points)
        coords = [coord for point in points for coord in (point.real, point.imag)]
        reversed_coords = [coord for point in reversed_points for coord in (point.real, point.imag)]

        if coords < reversed_coords:
            return coords + [width]
        else:
            return reversed_coords + [width]

    @abstractmethod
    def simplified(self, *args, **kwargs):
        ...

    @abstractmethod
    def svg_representation(self):
        ...

    @property
    def end_points(self):
        return (self.segment.start.real, self.segment.start.imag), (self.segment.end.real, self.segment.end.imag)

    @end_points.setter
    def end_points(self, end_points):
        self.segment.start = complex(*end_points[0])
        self.segment.end = complex(*end_points[1])

    start = property(lambda self: self.segment.start)
    end = property(lambda self: self.segment.end)


class Line(Primitive):
    def __init__(self, segment):
        self.segment = svgpathtools.Line(*(np.round(p, INTERNAL_PRECISION) for p in segment))

    @classmethod
    def from_points(cls, start, end):
        return cls(svgpathtools.Line(complex(*start), complex(*end)))

    def simplified(self, distinguishability_threshold):
        r'''Returns (None,) if end points coincide and (self,) otherwise.'''
        if pointsC_are_indistinguishable(self.segment.start, self.segment.end, distinguishability_threshold):
            return None,
        return self.copy(),

    def split_with_infline(self, l0, n):
        start, end = (np.array([p.real, p.imag]) for p in self.segment.bpoints())
        seg_l = end - start
        ldotn = seg_l.dot(n)
        if ldotn != 0:
            t = (l0 - start).dot(n) / ldotn
            if 0 <= t <= 1:
                intersection_point = start + seg_l * t
                return Line.from_points(start, intersection_point), Line.from_points(intersection_point, end)
        return self.copy(),

    def svg_representation(self):
        return (self.segment.start.real, self.segment.start.imag), (self.segment.end.real, self.segment.end.imag)


class _Bezier(Primitive):
    MAX_LINEAR_SEGMENTS = 2

    def simplified(self, distinguishability_threshold):
        r"""Returns
            1. (None,) if all control points coincide
            2. up to 3 Lines if their combination is "indistinguishable" from self when rendered
            3. (self,) otherwise
        """
        if all(pointsC_are_indistinguishable(self.segment[0], p, distinguishability_threshold) for p in
               self.segment[1:]):
            return None,

        if distinguishability_threshold == 0:
            return self.copy(),

        segments = simplify_bezier(self.segment, distinguishability_threshold,
                                   max_segments_n=self.__class__.MAX_LINEAR_SEGMENTS)
        if segments[0] == self.segment:
            return self.copy(),
        else:
            return (Line(line) for line in segments)

    def split_with_infline(self, l0, n):
        poly = self.segment.poly()
        poly_x = np.poly1d(poly.coefficients.real)
        poly_y = np.poly1d(poly.coefficients.imag)
        dist = (poly_x - l0[0]) * n[0] + (poly_y - l0[1]) * n[1]

        roots = sorted(set([0, 1] + [t.real for t in dist.roots if np.isreal(t) and (0 <= t <= 1)]))

        subsegs = [self.cropped(roots[i], roots[i + 1]) for i in range(len(roots) - 1)]
        return subsegs

    def svg_representation(self):
        return tuple((p.real, p.imag) for p in self.segment)


class CBezier(_Bezier):
    def __init__(self, segment):
        self.segment = svgpathtools.CubicBezier(*(np.round(p, INTERNAL_PRECISION) for p in segment))

    @classmethod
    def from_cubic(cls, segment):
        return cls(segment)

    @classmethod
    def from_quadratic(cls, segment):
        p0, p1, p2 = segment.bpoints()
        return cls(svgpathtools.CubicBezier(p0, p0 + (p1 - p0) * 2 / 3, p2 + (p1 - p2) * 2 / 3, p2))


class QBezier(_Bezier):
    def __init__(self, segment):
        self.segment = svgpathtools.QuadraticBezier(*(np.round(p, INTERNAL_PRECISION) for p in segment))

    @classmethod
    def from_cubic(cls, segment):
        p0, p1, p2, p3 = np.array([[p.real, p.imag] for p in segment.bpoints()])
        # if cubic curve is an elevated quadratic
        if np.all(np.round(p2 * 3 - p3, INTERNAL_PRECISION) == np.round(p1 * 3 - p0, INTERNAL_PRECISION)):
            return cls(svgpathtools.QuadraticBezier(p0, (p2 * 3 - p3) / 2, p3))
        else:  # split cubic in the inflection point
            pass # TODO

    @classmethod
    def from_quadratic(cls, segment):
        return cls(segment)


Bezier = CBezier  # our canonical bezier curves are cubic
#Bezier = QBezier  # our canonical bezier curves are quadratic
Primitive._seg2prim = {svgpathtools.Line: Line, svgpathtools.CubicBezier: Bezier.from_cubic,
                       svgpathtools.QuadraticBezier: Bezier.from_quadratic}
