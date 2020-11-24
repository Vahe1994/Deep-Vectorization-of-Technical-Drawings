from copy import copy, deepcopy

import svgpathtools

from .utils import parse
from .primitives import Primitive
from .utils.splitting import split_to_patches, crop_to_bbox
from . import units

from util_files.data.graphics_primitives import PT_LINE, PT_QBEZIER


class Path:
    def __init__(self, path, attributes, flatten_transforms=True, convert_to_pixels=True):
        # geometry
        if flatten_transforms:
            if 'transform' in attributes:
                matrix = svgpathtools.parser.parse_transform(attributes['transform'])
                path = svgpathtools.path.transform(path, matrix)

        self.segments = [primitive for seg in path if not isinstance(seg, svgpathtools.Arc)
                         for primitive in Primitive.from_seg(seg)]

        # appearance
        self.fill = parse.fill(attributes)
        self.width = parse.stroke(attributes)
        self.svg_attributes = {}
        self.is_control = False

        self.convert_to_pixels = convert_to_pixels
        if convert_to_pixels:
            if self.width is not None:
                self.width = self.width.as_pixels()
            # TODO convert segments to pixels?

    @classmethod
    def create_visible(cls, *args, **kwargs):
        path = cls(*args, **kwargs)
        if path.nonempty and path.visible:
            return path
        else:
            return None

    @classmethod
    def from_primitive(cls, primitive_type, primitive):
        if primitive_type == PT_LINE:
            start = primitive[0] + primitive[1] * 1j
            end = primitive[2] + primitive[3] * 1j
            width = primitive[4]
            primitive = svgpathtools.Line(start, end)
        elif primitive_type == PT_QBEZIER:
            p1 = primitive[0] + primitive[1] * 1j
            p2 = primitive[2] + primitive[3] * 1j
            p3 = primitive[4] + primitive[5] * 1j
            width = primitive[6]
            primitive = svgpathtools.QuadraticBezier(p1, p2, p3)
        else:
            raise NotImplementedError
        path = svgpathtools.Path(primitive)
        attributes = {'stroke': 'black', 'stroke-width': f'{width}', 'fill': 'none'}
        return cls(path, attributes)

    @classmethod
    def make_line(cls, p1, p2, width, color='black'):
        path = cls(
            svgpathtools.Path(
                svgpathtools.Line(p1[0] + p1[1] * 1j, p2[0] + p2[1] * 1j),
            ),
            {'stroke': color, 'stroke-width': f'{width}', 'fill': 'none'}
        )
        path.svg_attributes['stroke'] = color
        if isinstance(width, units.GraphicUnits):
            path.width = width
        return path

    @classmethod
    def make_rectangle(cls, left, top, bottom, right, width, color='black'):
        path = cls(
            svgpathtools.Path(
                svgpathtools.Line(left + top * 1j, right + top * 1j),
                svgpathtools.Line(right + top * 1j, right + bottom * 1j),
                svgpathtools.Line(right + bottom * 1j, left + bottom * 1j),
                svgpathtools.Line(left + bottom * 1j, left + top * 1j),
            ),
            {'stroke': color, 'stroke-width': f'{width}', 'fill': 'none'}
        )
        path.svg_attributes['stroke'] = color
        if isinstance(width, units.GraphicUnits):
            path.width = width
        return path

    def continuous_subpaths(self):
        if self.is_continuous():
            return [self]
        subpaths = []
        subpath_start = 0
        for i in range(len(self) - 1):
            if self[i].end != self.segments[i + 1].start:
                subpath = self.copy(segments=self[subpath_start: i + 1])
                subpaths.append(subpath)
                subpath_start = i + 1
        subpath = self.copy(segments=self[subpath_start: len(self)])
        subpaths.append(subpath)
        return subpaths

    def copy(self, segments=None):
        if segments is None:
            return deepcopy(self)
        else:
            self_segments = self.segments
            self.segments = []
            self_copy = deepcopy(self)
            self_copy.segments = segments
            self.segments = self_segments
            return self_copy

    def copy_shallow(self, segments=None):
        new_path = copy(self)
        if segments is not None:
            new_path.segments = list(segments)
        return new_path

    def crop(self, bbox):
        self.segments = list(filter(None, crop_to_bbox(self.segments, bbox)))
        return self

    def __eq__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        if (len(self) != len(other)) or (not self.has_same_attributes(other)):
            return False
        for s, o in zip(self.segments, other.segments):
            if not s == o:
                return False
        return True

    def __ne__(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        return not self == other

    def has_same_attributes(self, other):
        if not isinstance(other, Path):
            return NotImplemented
        if (self.fill != other.fill) or (self.width != other.width) or (self.svg_attributes != other.svg_attributes):
            return False
        return True

    def __getitem__(self, index):
        return self.segments[index]

    def is_continuous(self):
        return all(self[i].end == self[i + 1].start for i in range(len(self) - 1))

    def __iter__(self):
        return self.segments.__iter__()

    def __len__(self):
        return len(self.segments)

    def mirror(self, x):
        [seg.mirror(x) for seg in self.segments]

    def remove_fill(self, default_width):
        if self.fill:
            self.fill = False
            if self.width is None:
                self.width = units.fromrepr(default_width, units.Pixels)
                if self.convert_to_pixels:
                    self.width = self.width.as_pixels()

    def __repr__(self):
        return "Path({})".format(
            ",\n     ".join(repr(x) for x in self.segments))

    def rotate(self, rotation_deg, origin):
        [seg.rotate(rotation_deg, origin) for seg in self.segments]

    def scale(self, scale, only_coordinates=False):
        if (self.width is not None) and (not only_coordinates):
            self.width.scale(scale)
        [seg.scale(scale) for seg in self.segments]

    def scaled(self, scale):
        path = self.copy()
        path.scale(scale)
        return path

    def split_to_patches(self, *args, **kwargs):
        iS, jS, segments_in_patches = split_to_patches(self.segments, *args, **kwargs)
        return iS, jS, [self.copy_shallow(segments) for segments in segments_in_patches]

    def to_svgpathtools(self):
        path = svgpathtools.Path(*(seg.to_svgpathtools() for seg in self.segments))
        attributes = self.svg_attributes.copy()
        if 'fill' not in attributes:
            attributes['fill'] = 'black' if self.fill else 'none'
        if self.width is not None:
            if 'stroke' not in attributes:
                attributes['stroke'] = 'black'
            if 'stroke-width' not in attributes:
                attributes['stroke-width'] = str(self.width)
        return path, attributes

    def translate(self, t):
        [seg.translate(t) for seg in self.segments]

    def translated(self, t):
        path = self.copy()
        path.translate(t)
        return path

    def with_removed_tiny_segments(self, threshold):
        self.segments = list(
            filter(None, (subseg for seg in self.segments for subseg in seg.collapsed_if_tiny(threshold))))
        return self

    def with_simplified_segments(self, distinguishability_threshold):
        self.segments = list(
            filter(None, (subseg for seg in self.segments for subseg in seg.simplified(distinguishability_threshold))))
        return self

    @property
    def end_points(self):
        return (self[0].start.real, self[0].start.imag), (self[-1].end.real, self[-1].end.imag)

    nonempty = property(lambda self: len(self.segments) > 0)
    visible = property(lambda self: self.width is not None or self.fill)
