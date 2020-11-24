from svgpathtools import Arc
from svgpathtools import CubicBezier
from svgpathtools import Line
from svgpathtools import Path
from svgpathtools import polygon
from svgpathtools import QuadraticBezier
import svgpathtools

from . import transforms


def adjust_path_ppi(path, attributes, new_ppi_to_old_ppi, scale_width=1):
    if new_ppi_to_old_ppi != 1:
        path = path.scaled(new_ppi_to_old_ppi)
    if 'stroke-width' in attributes:
        attributes['stroke-width'] = str(float(attributes['stroke-width']) * new_ppi_to_old_ppi * scale_width)
    return path, attributes


def clean_attributes(attributes):
    # leave only appearance attributes
    new_attributes = dict()
    for key in set(['fill', 'fill-opacity',
                    'stroke', 'stroke-width', 'stroke-linecap', 'stroke-linejoin', 'stroke-opacity', 'stroke-miterlimit']
                  ).intersection(attributes.keys()):
        new_attributes[key] = attributes[key]
    
    return new_attributes


def flatten_transforms(paths, attribute_dicts):
    paths_out = []
    attributes_out = []
    
    for path, attributes in zip(paths, attribute_dicts):
        if 'transform' in attributes:
            matrix = svgpathtools.parser.parse_transform(attributes['transform'])
            path = svgpathtools.path.transform(path, matrix)
            del attributes['transform']

        paths_out.append(path)
        attributes_out.append(attributes)            

    return paths_out, attributes_out            


def get_graphics_in_rect(paths, attribute_dicts, boundary):
    paths_out = []
    attributes_out = []
    
    left, top, right, bottom = boundary
    boundary_rect = polygon(left + 1j * top, right + 1j * top, right + 1j * bottom, left+ 1j * bottom)

    for path, attributes in zip(paths, attribute_dicts):
        new_path = Path()
        for seg in path:
            # the line is inside the patch if it coincides with one of the edges
            # svgpathtools assumes that two segments do not coinside when it searches for intersections so we need to address this case separately
            if isinstance(seg, Line):
                if any(seg == edge for edge in boundary_rect):
                    new_path.append(seg)
                    continue

            # for a given segment find all its intersections with the boundary
            intersections = []
            for edge in boundary_rect:
                intersections = intersections + seg.intersect(edge)
            
            # if svgpathtools found no intersections then there are 3 cases
            if len(intersections) == 0:
                seg_is_in_patch = point_is_in_rect(seg.start, boundary)
                # the segment lies completely inside the patch
                if seg_is_in_patch == 1:
                    new_path.append(seg)
                    continue
                # the segment lies completely outside the patch
                elif seg_is_in_patch == -1:
                    continue
                # the segment is line and it lies inside on of the edges of the patch
                else:
                    new_path.append(seg)
                    continue
            else:
                # minmax the intersection point since sometimes it lies slightly outside of the segment
                split_points = [max(min(pair[0], 1), 0) for pair in intersections]
                split_points.append(0)
                split_points.append(1)
                split_points = sorted(set(split_points))
                # split segment in the intersection points
                for i in range(len(split_points) - 1):
                    subseg = seg.cropped(split_points[i], split_points[i+1])
                    
                    start_is_in_patch = point_is_in_rect(subseg.start, boundary)
                    end_is_in_patch = point_is_in_rect(subseg.end, boundary)
                    
                    # for each type of segment if one of the endpoints lies outside of the patch then the segment is outside
                    if start_is_in_patch == -1 or end_is_in_patch == -1:
                        continue

                    # else
                    # the lines are either on the boundary or inside
                    if isinstance(subseg, Line):
                        new_path.append(subseg)
                        continue

                    # all the other types -- Arc, QuadraticBezier, CubicBezier -- have no more common points with the boundary other than the endpoints, so any inner point of the segment is either inside or outside, which corresponds to the insideness of the whole segment
                    # the only exception is when the patch boundary is tangent to the segment in an inner point, in which case this point isn't considered as intersection point by svgpathtools.whatsoever, in this case the segment is inside the patch
                    if point_is_in_rect(subseg.point(.5), boundary) >= 0:
                        new_path.append(subseg)
                        continue

        if len(new_path) > 0:
            paths_out.append(new_path)
            attributes_out.append(attributes)            

    return paths_out, attributes_out


def outline_filled(attributes, outline_style=None):
    if path_is_filled(attributes):
        attributes['fill'] = 'none'
        if not path_is_outlined(attributes):
            for key in outline_style.keys():
                attributes[key] = outline_style[key]

    return attributes


def path_is_filled(attributes):
    return not ('fill' in attributes and attributes['fill'] == 'none')


def path_is_outlined(attributes):
    return 'stroke' in attributes and attributes['stroke'] != 'none' and (not 'stroke-width' in attributes or attributes['stroke-width'] != '0')


def path_is_invisible(attributes):
    return not path_is_filled(attributes) and not path_is_outlined(attributes)


# this function may not work properly for points close to the edge since svgpathtools sometimes finds intersections of the segments that lie slightly off
def point_is_in_rect(point, boundary):
    '''Return 1 if the point is inside, -1 if it's outside, and 0 if it's on the boundary.'''
    
    left, top, right, bottom = boundary
    x, y = point.real, point.imag
    
    if x < left or x > right or y < top or y > bottom:
        return -1
    if left < x < right:
        if top < y < bottom:
            return 1
        else:
            return 0
    else:
        return 0


def remove_empty_segments(path):
    return Path(*(seg for seg in path if segment_is_not_empty(seg)))


def remove_empty_paths(paths, attribute_dicts, remove_filled=False):
    paths_out = []
    attributes_out = []

    for path, attributes in zip(paths, attribute_dicts):
        if remove_filled:
            if path_is_filled(attributes):
                continue
        if path_is_invisible(attributes):
            continue
        path = remove_empty_segments(path)
        if len(path) > 0:
            paths_out.append(path)
            attributes_out.append(attributes)
        
    return paths_out, attributes_out


def segment_is_not_empty(seg):
    if isinstance(seg, (Line, Arc)):
        return seg.start != seg.end
    if isinstance(seg, (CubicBezier, QuadraticBezier)):
        return any(p != seg[0] for p in seg)
    return True
