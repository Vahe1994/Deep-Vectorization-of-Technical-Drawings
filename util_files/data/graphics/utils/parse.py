import css_parser as cssutils

from util_files.data.graphics import units
from util_files import warnings


throw_transparency_warning = lambda attributes='': warnings.warn('Transparency is ignored. {}'.format(attributes), warnings.UndefinedWarning)


def fill(attributes):
    '''Returns True for any visible filling.'''
    if 'fill' not in attributes: # default is to fill with black
        return True

    color = paint(attributes['fill'])
    if not color:
        return False

    if 'fill-opacity' in attributes:
        opacity = cssutils.css.DimensionValue(attributes['fill-opacity'])
        if opacity.dimension == '%':
            opacity = opacity.value / 100
        else:
            opacity = opacity.value
        if opacity == 0:
            return False
        elif opacity != 1:
            throw_transparency_warning()

    if 'fill-rule' in attributes and attributes['fill-rule'] != 'evenodd':
        warnings.warn('Filling rule "evenodd" are not implemented: {}'.format(attributes), warnings.UndefinedWarning)

    return True


def paint(s):
    '''Returns False for 'none' or fully transparent and True otherwise.'''
    if s.lower() == 'none':
        return False

    value = cssutils.css.ColorValue(s)
    if value.alpha == 0:
        return False
    elif value.alpha != 1:
        throw_transparency_warning()
    if value.red == 255 and value.green == 255 and value.blue == 255:
        warnings.warn('White color is converted to black.', warnings.UndefinedWarning)

    return True


def stroke(attributes):
    '''Returns width of the stroke for visible strokes and None otherwise.'''
    if 'stroke' not in attributes:
        return None

    color = paint(attributes['stroke'])
    if not color:
        return None

    if 'stroke-opacity' in attributes:
        opacity = cssutils.css.DimensionValue(attributes['stroke-opacity'])
        if opacity.dimension == '%':
            opacity = opacity.value / 100
        else:
            opacity = opacity.value
        if opacity == 0:
            return None
        elif opacity != 1:
            throw_transparency_warning()

    if 'stroke-width' in attributes:
        width = units.fromrepr(attributes['stroke-width'], units.Pixels)
        if width.value == 0:
            return None
    else:
        width = units.fromrepr('1', units.Pixels)

    if 'stroke-linecap' in attributes and attributes['stroke-linecap'] != 'square':
        warnings.warn('Different linecaps are not supported.', warnings.UndefinedWarning)
    if 'stroke-linejoin' in attributes and attributes['stroke-linejoin'] != 'miter':
        warnings.warn('Different linejoins are not supported.', warnings.UndefinedWarning)
    if 'stroke-dasharray' in attributes:
        warnings.warn('Dashed strokes are not supported.', warnings.UndefinedWarning)
    if 'stroke-miterlimit' in attributes:
        warnings.warn('Miterlimit is ignored.', warnings.UndefinedWarning)

    return width


def split_path_style_to_attributes(style_str):
    if style_str[-1] == ';':
        style_str = style_str[:-1]
    style_attributes = (pair.split(':') for pair in style_str.split(';'))
    return {kv[0]: kv[1] for kv in style_attributes}
