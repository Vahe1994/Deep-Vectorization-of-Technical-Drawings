from sys import byteorder

import cairocffi as cairo
import numpy as np

from util_files.data.graphics_primitives import PT_LINE, PT_ARC, PT_CBEZIER, PT_QBEZIER, PT_QBEZIER_B, PT_POINT
from util_files.data.graphics.primitives import CBezier, Line, QBezier
from .skeleton import make_skeleton_vahe
from .utils import qbezier_to_cbezier

def render(data, dimensions, data_representation='paths', linecaps='square', linejoin='miter'):
    # prepare data buffer
    width, height = dimensions
    buffer_width = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, width)
    image = np.ndarray(shape=(height, buffer_width), dtype=np.uint8).ravel()

    # get drawing parameters
    linecaps = _linecaps[linecaps]
    linejoin = _linejoin[linejoin]

    # draw

    surface = cairo.ImageSurface(cairo.FORMAT_A8, *dimensions, data=memoryview(image), stride=buffer_width)
    with cairo.Context(surface) as ctx:
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_line_join(linejoin)
        ctx.set_line_cap(linecaps)

        # ignore r,g,b -- draw in alpha
        set_color_to_bg = lambda: ctx.set_source_rgba(0, 0, 0, 1) # white
        set_color_to_fg = lambda: ctx.set_source_rgba(0, 0, 0, 0) # black

        # fill bg
        set_color_to_bg()
        ctx.paint()

        # draw
        set_color_to_fg()
        _render_data[data_representation](data, ctx)

    # remove int32 padding
    image = image.reshape(height, buffer_width)[:, :width]

    return image


def render_with_skeleton(data, dimensions, data_representation='vahe', line_width=1, node_size=4, control_line_width=.5, control_node_size=2, line_color=(1,1,0), node_color=(1,0,0), controls_color=(0,0,1), linecaps='square', linejoin='miter'):
    if data_representation != 'vahe':
        raise NotImplementedError
    # prepare data buffer
    width, height = dimensions
    buffer_width = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_RGB24, width)
    image = np.ndarray(shape=(height, buffer_width), dtype=np.uint8).ravel()

    # get drawing parameters
    linecaps = _linecaps[linecaps]
    linejoin = _linejoin[linejoin]

    # draw
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, *dimensions, data=memoryview(image), stride=buffer_width)

    if byteorder == 'little':
        rgb_slice = slice(0,3)
        line_color = reversed(line_color)
        node_color = reversed(node_color)
        controls_color = reversed(controls_color)
    else:
        rgb_slice = slice(1,4)
    with cairo.Context(surface) as ctx:
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_line_join(linejoin)
        ctx.set_line_cap(linecaps)

        # ignore r,g,b -- draw in alpha
        set_color_to_bg = lambda: ctx.set_source_rgb(1, 1, 1) # white
        set_color_to_fg = lambda: ctx.set_source_rgb(0, 0, 0) # black

        # fill bg
        set_color_to_bg()
        ctx.paint()

        # draw
        set_color_to_fg()
        _render_data[data_representation](data, ctx)

        # draw skeleton
        edges, nodes, controls = _make_skeleton[data_representation](data, line_width=line_width, node_size=node_size, control_line_width=control_line_width, control_node_size=control_node_size)

        ctx.set_source_rgb(*line_color)
        _render_data[data_representation](edges, ctx)

        ctx.set_source_rgb(*node_color)
        _render_data[data_representation](nodes, ctx)

        ctx.set_source_rgb(*controls_color)
        _render_data[data_representation](controls, ctx)

    # remove alpha padding
    image = image.reshape(height, -1, 4)[:, :width, rgb_slice]

    return image


def render_paths(paths, ctx):
    for path in paths:
        with ctx:
            ctx.new_path()
            for primitive in path:
                _draw_prim[primitive.__class__](ctx, *primitive.svg_representation())
            ctx.set_line_width(path.width)
            ctx.stroke()


def render_vahe(primitive_sets, ctx):
    with ctx:
        for primitive_type, primitives in primitive_sets.items():
            draw_function = _draw_repr[primitive_type]
            for primitive in primitives:
                draw_function(ctx, primitive)


def draw_line(ctx, p1, p2):
    ctx.move_to(*p1)
    ctx.line_to(*p2)


def draw_bezier(ctx, p1, p2, p3, p4):
    ctx.move_to(*p1)
    ctx.curve_to(*p2, *p3, *p4)


def draw_qbezier(ctx, p1, p2, p3):
    bezier = qbezier_to_cbezier(np.concatenate([p1, p2, p3]))
    draw_bezier(ctx, bezier[:2], bezier[2:4], bezier[4:6], bezier[6:8])


def draw_line_vahe(ctx, line):
    ctx.move_to(*line[:2])
    ctx.line_to(*line[2:4])
    ctx.set_line_width(line[4])
    ctx.stroke()


def draw_bezier_vahe(ctx, bezier):
    ctx.move_to(*bezier[:2])
    ctx.curve_to(*bezier[2:8])
    ctx.set_line_width(bezier[8])
    ctx.stroke()


def draw_qbezier_vahe(ctx, bezier):
    bezier = qbezier_to_cbezier(bezier)
    draw_bezier_vahe(ctx, bezier)


def draw_point_vahe(ctx, point):
    ctx.arc(*point[:2], point[2]/2, 0, np.pi * 2)
    ctx.fill()


_draw_prim = {Line: draw_line, CBezier: draw_bezier, QBezier: draw_qbezier}
_draw_repr = {PT_LINE: draw_line_vahe, PT_CBEZIER: draw_bezier_vahe, PT_QBEZIER: draw_qbezier_vahe,
              PT_QBEZIER_B: draw_qbezier_vahe, PT_POINT: draw_point_vahe}
_render_data = {'paths': render_paths, 'vahe': render_vahe}
_make_skeleton = {'vahe': make_skeleton_vahe}
_linecaps = {'square': cairo.LINE_CAP_SQUARE, 'butt': cairo.LINE_CAP_BUTT, 'round': cairo.LINE_CAP_ROUND}
_linejoin = {'miter': cairo.LINE_JOIN_MITER}
