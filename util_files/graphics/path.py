from copy import copy, deepcopy

import svgpathtools

from .utils import parse
from .primitives import Primitive
from .utils.splitting import split_to_patches, crop_to_bbox
from . import units


class Path:
    def __init__(self, path, attributes, flatten_transforms=True, convert_to_pixels=True):
        # geometry
        if flatten_transforms:
            if 'transform' in attributes:
                matrix = svgpathtools.parser.parse_transform(attributes['transform'])
                path = svgpathtools.path.transform(path, matrix)

        self.segments = [Primitive.from_seg(seg) for seg in path if not isinstance(seg, svgpathtools.Arc)]

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

    def copy(self):
        return deepcopy(self)

    def copy_shallow(self, segments=None):
        new_path = copy(self)
        if segments is not None:
            new_path.segments = list(segments)
        return new_path

    def crop(self, bbox):
        self.segments = list(filter(None, crop_to_bbox(self.segments, bbox)))
        return self

    def __getitem__(self, index):
        return self.segments[index]

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

    def scale(self, scale):
        if self.width is not None:
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

    def with_simplified_segments(self, distinguishability_threshold):
        self.segments = list(
            filter(None, (subseg for seg in self.segments for subseg in seg.simplified(distinguishability_threshold))))
        return self

    nonempty = property(lambda self: len(self.segments) > 0)
    visible = property(lambda self: self.width is not None or self.fill)
