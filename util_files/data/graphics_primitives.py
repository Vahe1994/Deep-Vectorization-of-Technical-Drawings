from enum import Enum, auto


# from vectran.renderers.cairo import cairo_line, cairo_bezier, cairo_arc
from util_files.geometric import liang_barsky_screen


class PrimitiveType(Enum):
    PT_LINE = auto()
    PT_CBEZIER = auto()
    PT_ARC = auto()
    PT_POINT = auto()
    PT_QBEZIER = auto()
    PT_QBEZIER_B = auto()


PT_ARC = PrimitiveType.PT_ARC
PT_CBEZIER = PrimitiveType.PT_CBEZIER
PT_QBEZIER = PrimitiveType.PT_QBEZIER
PT_QBEZIER_B = PrimitiveType.PT_QBEZIER_B
PT_LINE = PrimitiveType.PT_LINE
PT_POINT = PrimitiveType.PT_POINT


repr_len_by_type = {
    PT_LINE: 5,
    PT_CBEZIER: 9,
    PT_QBEZIER: 7,
    PT_QBEZIER_B: 7,
    PT_ARC: 6,
}


class GraphicsPrimitive(object):
    def __init__(self):
        self.is_drawn = True

    def draw(self, ctx): raise NotImplementedError

    @classmethod
    def from_repr(cls, line_repr): raise NotImplementedError

    def to_repr(self): raise NotImplementedError

    def clip_to_box(self, ctx): raise NotImplementedError


class Line(GraphicsPrimitive):
    def __init__(self, point1, point2, width):
        self.point1 = point1
        self.point2 = point2
        self.width = width
        super().__init__()

    def draw(self, ctx):
        '''cairo_line(ctx, self.to_repr())'''

    def clip_to_box(self, box_size):
        width, height = box_size
        bbox = (0, 0, width, height)
        clipped_point1, clipped_point2, self.is_drawn = \
            liang_barsky_screen(self.point1, self.point2, bbox)
        if self.is_drawn:
            self.point1, self.point2 = clipped_point1, clipped_point2

    @classmethod
    def from_repr(cls, line_repr):
        assert len(line_repr) == repr_len_by_type[PrimitiveType.PT_LINE]
        return cls(tuple(line_repr[0:2]),
                   tuple(line_repr[2:4]), line_repr[4])

    def to_repr(self):
        if self.point1 < self.point2:
            return self.point1 + self.point2 + (self.width,)
        else:
            return self.point2 + self.point1 + (self.width,)

    def to_svg(self):
        from svgpathtools import Line as SvgLine
        return SvgLine(complex(*self.point1), complex(*self.point2))


class BezierCurve(GraphicsPrimitive):
    def __init__(self, cpoint1, cpoint2, cpoint3, cpoint4, width):
        self.cpoint1 = cpoint1
        self.cpoint2 = cpoint2
        self.cpoint3 = cpoint3
        self.cpoint4 = cpoint4
        self.width = width
        super().__init__()

    def draw(self, ctx):
        '''return cairo_bezier(ctx, self.to_repr())'''

    def clip_to_box(self, box_size):
        raise NotImplementedError

    @classmethod
    def from_repr(cls, bezier_repr):
        assert len(bezier_repr) == repr_len_by_type[PrimitiveType.PT_BEZIER]
        return cls(tuple(bezier_repr[0:2]),
                   tuple(bezier_repr[2:4]),
                   tuple(bezier_repr[4:6]),
                   tuple(bezier_repr[6:8]),
                   bezier_repr[8])

    def to_repr(self):
        cpoints_direct = (self.cpoint1, self.cpoint2, self.cpoint3, self.cpoint4)
        cpoints_inverse = tuple(coord for point in reversed(cpoints_direct) for coord in point)
        cpoints_direct = tuple(coord for point in cpoints_direct for coord in point)
        if cpoints_direct < cpoints_inverse:
            return cpoints_direct + (self.width, )
        else:
            return cpoints_inverse + (self.width, )


class Arc(GraphicsPrimitive):
    def __init__(self, center, radius, angle1, angle2, width):
        self.center = center
        self.radius = radius
        self.angle1 = angle1
        self.angle2 = angle2
        self.width = width
        super().__init__()

    def draw(self, ctx):
        '''return cairo_arc(ctx, self.to_repr())'''

    def clip_to_box(self, box_size):
        raise NotImplementedError

    @classmethod
    def from_repr(cls, arc_repr):
        assert len(arc_repr) == repr_len_by_type[PrimitiveType.PT_ARC]
        return cls(tuple(arc_repr[0:2]),
                   arc_repr[2],
                   arc_repr[3],
                   arc_repr[4],
                   arc_repr[5])

    def to_repr(self):
        return self.center + (self.radius, self.angle1, self.angle2, self.width)


__all__ = [
    'PT_LINE',
    'PT_ARC',
    'PT_QBEZIER',
    'PT_CBEZIER',
    'PT_POINT',
    'Line'
]
