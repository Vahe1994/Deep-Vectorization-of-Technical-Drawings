#!/usr/bin/env python3
import argparse

from .utils.render_utils import read_step_file, export_shape_to_svg


def render_step_file(args):
    shape = read_step_file(args.input_file, return_as_shapes=False)
    export_shape_to_svg(
        shape, args.output_file,
        width=args.width, height=args.height,
        margin_left=args.margin_left, margin_top=args.margin_top,
        export_hidden_edges=args.export_hidden_edges,
        location=args.location, direction=args.direction,
        line_width=args.line_width
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', dest='input_file',
                        required=True, help='input dir with ABC dataset.')
    parser.add_argument('-o', '--output-file', dest='output_file',
                        required=True, help='output file.')
    parser.add_argument('--width', default=500, dest='width', help='render width.')
    parser.add_argument('--height', default=500, dest='height', help='render height.')
    parser.add_argument('--line-width', default=0.5, dest='line_width')
    parser.add_argument('--margin-top', default=30, dest='margin_top', help='render margin on top.')
    parser.add_argument('--margin-left', default=30, dest='margin_left', help='render margin on left.')
    parser.add_argument('--loc', default=[0, 0, 0], dest='location', help='projection hyperplane normal origin')
    parser.add_argument('--dir', default=[1, 1, 1], dest='direction', help='projection hyperplane normal direction')
    parser.add_argument('--export-hidden-edges', dest='export_hidden_edges', action='store_true',
                        help='render hidden edges as dashed lines')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    render_step_file(args)
