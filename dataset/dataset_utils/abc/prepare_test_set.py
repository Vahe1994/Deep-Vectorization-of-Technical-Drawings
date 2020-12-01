from glob import glob
import sys

from PIL import Image
import numpy as np

sys.path.append('/code')
from util_files.simplification.join_qb import join_quad_beziers
from util_files.rendering.cairo import PT_QBEZIER, render
from util_files.data.graphics.graphics import Path, VectorImage


def prepare_image(filename, width, tiny_segments_threshold=1.5, quad_beziers_fit_tol=.1, quad_beziers_w_tol=.1, max_relative_overlap=.5):
    vector_image = VectorImage.from_svg(filename)

    # 1. Choose random width
    vector_image.scale_to_width('min', width)

    # 2. Remove short segments
    vector_image.remove_tiny_segments(tiny_segments_threshold)

    # 3. Simplify curves
    lines, curves = vector_image.vahe_representation()
    lines = np.asarray(lines)
    if len(curves) > 0:
        curves = join_quad_beziers(curves, fit_tol=quad_beziers_fit_tol, w_tol=quad_beziers_w_tol).numpy()
    curves = np.asarray(curves)

    # 4. Convert lines to curves
    curves = curves.tolist() + [line_to_curve(line) for line in lines]
    vector_image.paths = ([Path.from_primitive(PT_QBEZIER, prim) for prim in curves])
    return vector_image


def line_to_curve(line):
    p1 = line[:2]
    p3 = line[2:4]
    p2 = (p1 + p3) / 2
    width = line[4]
    return np.array([*p1, *p2, *p3, width])


if __name__ == '__main__':
    input_dir = '/data/svg_datasets/whole_images/abc/test'
    output_dir = '/data/test_sets/abc'

    width_min = 1
    width_max = 7
    np.random.seed(638)

    for filename in glob(f'{input_dir}/*.svg'):
        sample = filename[len(input_dir)+1:-4]
        width = np.random.random() * (width_max - width_min) + width_min

        print(f'Processing {sample}: width is {width}')
        vector_image = prepare_image(filename, width)
        vector_image.save(f'{output_dir}/{sample}.svg')
        Image.fromarray(vector_image.render(render)).save(f'{output_dir}/{sample}.png')
