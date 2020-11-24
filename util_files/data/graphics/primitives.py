from abc import ABC, abstractmethod
from copy import copy, deepcopy

from scipy.optimize import minimize
import numpy as np
import svgpathtools

from .utils.common import mirror_point_x
from util_files.simplification.utils import pointsC_are_indistinguishable
from util_files.simplification.simplify import bezier as simplify_bezier
from util_files import warnings

INTERNAL_PRECISION = 4
_cubic_to_quad = None


def cubic_to_quad(*args, maximum_allowed_distance=1):
    global _cubic_to_quad
    if _cubic_to_quad is None:
        import js2py
        _cubic_to_quad = js2py.require('cubic2quad')
    return _cubic_to_quad(*args, maximum_allowed_distance)


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

    def __eq__(self, other):
        if not isinstance(other, Primitive):
            return NotImplemented
        return self.segment == other.segment

    def __ne__(self, other):
        if not isinstance(other, Primitive):
            return NotImplemented
        return not self == other

    def distance_to(self, other):
        if not isinstance(other, Primitive):
            return NotImplemented

        def dist(params):
            t_self, t_other = params
            return abs(self.segment.poly()(t_self) - other.segment.poly()(t_other))

        res = minimize(dist, [.5, .5], bounds=[[0, 1], [0, 1]])
        if not res.success:
            warnings.warn('Minimization of distance failed', warnings.UndefinedWarning)
        return res.fun

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
    def collapsed_if_tiny(self, *args, **kwargs):
        ...

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

    @classmethod
    def from_line(cls, segment):
        return [cls(segment)]

    def collapsed_if_tiny(self, distinguishability_threshold):
        r'''Returns (None,) if end points coincide and (self,) otherwise.'''
        if pointsC_are_indistinguishable(self.segment.start, self.segment.end, distinguishability_threshold):
            return None,
        return self.copy(),

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

    def collapsed_if_tiny(self, distinguishability_threshold):
        if all(pointsC_are_indistinguishable(self.segment[0], p, distinguishability_threshold) for p in
               self.segment[1:]):
            return None,
        return self.copy(),

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
        return [cls(segment)]

    @classmethod
    def from_quadratic(cls, segment):
        p0, p1, p2 = segment.bpoints()
        return [cls(svgpathtools.CubicBezier(p0, p0 + (p1 - p0) * 2 / 3, p2 + (p1 - p2) * 2 / 3, p2))]


class QBezier(_Bezier):
    def __init__(self, segment):
        self.segment = svgpathtools.QuadraticBezier(*(np.round(p, INTERNAL_PRECISION) for p in segment))

    @classmethod
    def from_cubic(cls, segment):
        ccoords = [coord for point in segment.bpoints() for coord in (point.real, point.imag)]
        qcoords = cubic_to_quad(*ccoords)
        qpoints = [qcoords[i] + qcoords[i + 1] * 1j for i in range(0, len(qcoords), 2)]
        quads = [qpoints[i:i + 3] for i in range(0, len(qpoints) - 1, 2)]
        return [cls(svgpathtools.QuadraticBezier(*quad)) for quad in quads]

    @classmethod
    def from_quadratic(cls, segment):
        return [cls(segment)]


# Bezier = CBezier  # our canonical bezier curves are cubic
Bezier = QBezier  # our canonical bezier curves are quadratic
Primitive._seg2prim = {svgpathtools.Line: Line.from_line, svgpathtools.CubicBezier: Bezier.from_cubic,
                       svgpathtools.QuadraticBezier: Bezier.from_quadratic}
