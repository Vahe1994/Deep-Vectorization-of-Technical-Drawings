from glob import glob

import svgpathtools


def clean_paths(paths_in, attributes_in, outline_filled=True):
    paths_cleaned = []
    attributes_cleaned = []

    for path, attributes in zip(paths_in, attributes_in):
        path = clean_path(path)
        if path is None:
            continue

        attrubutes = clean_attributes(attributes, outline_filled=outline_filled)
        
        paths_cleaned.append(path)
        attributes_cleaned.append(attributes)
        
    return paths_cleaned, attributes_cleaned


def clean_path(path):
    # remove empty paths
    if len(path) == 0:
        return None

    # remove empty segments
    empty_segments = []
    for seg in path:
        if seg.start == seg.end:
            empty_segments.append(seg)
    for seg in empty_segments:
        if len(path) > 1:
            path.remove(seg)
        else:
            return None
    return path


def clean_attributes(attributes, outline_filled=True, outline_style=None):
    # remove unneded attributes since everything is Path now; basically, leave only appearance attributes
    unneded_attributes = ('x', 'y', 'width', 'height', 'rx', 'ry', # rect, ellispse
                          'r', 'cx', 'cy', # circle, ellispse
                          'x1', 'x2', 'y1', 'y2', # line
                          'points', # polyline, polygon
                          'd', # path
                          'fill-rule') # filling attributes
    for key in unneded_attributes:
        if key in attributes: del attributes[key]

    if outline_filled:
        if 'fill' in attributes and attributes['fill'] == 'none':
            pass
        else:
            attributes['fill'] = 'none'
            if outline_style is None:
                # TODO implement random appearance selection
                # skip path if it already has outline
                if 'stroke' in attributes and attributes['stroke'] != 'none':
                    pass
                else:
                    attributes['stroke'] = '#000000'
                    attributes['stroke-linecap'] = 'round'
                    attributes['stroke-linejoin'] = 'round'
                    attributes['stroke-miterlimit'] = '10'
                    attributes['stroke-width'] = '7'
            else:
                assert False, 'Not implemented'

    return attributes


for ground_svg in glob('*/ground.svg'):
    root_dir = '/'.join(ground_svg.split('/')[:-1])
    print(root_dir)

    ground_paths, ground_attributes = svgpathtools.svg2paths(ground_svg)
    ground_paths, ground_attributes = clean_paths(ground_paths, ground_attributes)

    for file_svg in glob(root_dir + '/file*.svg'):
        print(file_svg)
        paths, attribute_dicts, svg_attributes = svgpathtools.svg2paths2(file_svg)
        paths, attribute_dicts = clean_paths(paths, attribute_dicts)
        paths = paths + ground_paths
        attribute_dicts = attribute_dicts + ground_attributes
        svgpathtools.wsvg(paths, filename=file_svg, attributes=attribute_dicts, svg_attributes=svg_attributes)
