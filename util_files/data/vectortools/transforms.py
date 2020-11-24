import random

from svgpathtools import Arc
from svgpathtools import CubicBezier
from svgpathtools import Line
from svgpathtools import Path
from svgpathtools import QuadraticBezier


def random_jitter(paths, translation_amplitude=0j, radius_amplitude=0, rotation_amplitude=0):
    """Displace vector graphics randomly.

    :param paths: paths sequence
    :type paths: iterable over `Path`
    :param translation_amplitude: amplitude of points translation
    :type translation_amplitude: complex, real part corresponds to horizontal and imaginary -- to vertical translations 
    :param radius_amplitude: amplitude of relative arc radii transformation; radius is scaled by `random.uniform(1 - radius_amplitude, 1 + radius_amplitude)`
    :param rotation_amplitude: amplitude of arc rotation transformation
    """
    # gather all points to transform the coinciding points accordingly
    points = set()
    radii = set()
    for path in paths:
        for seg in path:
            if isinstance(seg, (Line, QuadraticBezier, CubicBezier)):
                points.update(seg.bpoints())
            elif isinstance(seg, Arc):
                points.add(seg.start)
                points.add(seg.end)
                radii.add(seg.radius)
            else:
                assert False, 'Unknown segment type {}'.format(seg.__class__)
                
    # transform all points accordingly
    transformed_points = {point : point + random.uniform(-translation_amplitude.real, translation_amplitude.real) + random.uniform(-translation_amplitude.imag, translation_amplitude.imag) * 1j for point in points}
    transformed_radii = {radius : radius.real * random.uniform(1 - radius_amplitude, 1 + radius_amplitude) + radius.imag * random.uniform(1 - radius_amplitude, 1 + radius_amplitude) * 1j for radius in radii}
    
    # transform data
    def transform_path(path):
        return Path(*[transform_seg(seg) for seg in path])
    
    def transform_seg(seg):
        if isinstance(seg, (Line, QuadraticBezier, CubicBezier)):
            return seg.__class__(*[transformed_points[point] for point in seg.bpoints()])
        elif isinstance(seg, Arc):
            new_start = transformed_points[seg.start]
            new_end = transformed_points[seg.end]
            new_raidus = transformed_radii[seg.radius]
            new_rotation = seg.rotation + random.uniform(-rotation_amplitude, rotation_amplitude)
            return Arc(new_start, new_raidus, new_rotation, seg.large_arc, seg.sweep, new_end, seg.autoscale_radius)
    
    return [transform_path(path) for path in paths]


def random_rotate(paths, rotation_amplitude):
    """Rotate each path randomly around its center.

    :param paths: paths sequence
    :type paths: iterable over `Path`
    :param deg_amplitude: rotate by `random.uniform(-rotation_amplitude, rotation_amplitude)` degrees
    """
    bbox_to_center = lambda xmin, xmax, ymin, ymax: (xmin + xmax + (ymin + ymax) * 1j) / 2 
    return [path.rotated(random.uniform(-rotation_amplitude, rotation_amplitude), bbox_to_center(*path.bbox())) for path in paths]


def random_translate(paths, translation_amplitude):
    """Translate each path randomly.

    :param paths: paths sequence
    :type paths: iterable over `Path`
    :param translation_amplitude: amplitude of points translation
    :type translation_amplitude: complex, real part corresponds to horizontal and imaginary -- to vertical translations
    """
    return [path.translated(random.uniform(-translation_amplitude.real, translation_amplitude.real) + random.uniform(-translation_amplitude.imag, translation_amplitude.imag)*1j) for path in paths]


def rotate(paths, deg, origin):
    """Rotate everyting.

    :param paths: paths sequence
    :type paths: iterable over `Path`
    :param deg: rotate by `deg` degrees
    :param origin: rotate around `origin`
    """
    return [path.rotated(deg, origin) for path in paths]


def translate(paths, t):
    """Translate everyting.

    :param paths: paths sequence
    :type paths: iterable over `Path`
    :param t: translation vector
    :type t: complex, real part corresponds to horizontal and imaginary -- to vertical translations 
    """
    return [path.translated(t) for path in paths]
