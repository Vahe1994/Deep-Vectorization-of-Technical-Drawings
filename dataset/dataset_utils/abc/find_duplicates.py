from glob import glob
import os
import sys

import numpy as np

sys.path.append('/code')
from util_files.data.graphics.graphics import VectorImage
from util_files.rendering.cairo import render


def find_duplicates(filepath, others_paths, max_iou=.9):
    image = VectorImage.from_svg(filepath)
    rendering_binarized = image.render(render) < 255 / 2
    width = image.width
    height = image.height

    for other_path in others_paths:
        if other_path == filepath:
            continue
        other_image = VectorImage.from_svg(other_path)
        if (other_image.height != height) or (other_image.width != width):
            continue
        other_rendering = other_image.render(render) < 255 / 2
        iou = (rendering_binarized & other_rendering).sum() / (rendering_binarized | other_rendering).sum()
        if iou > max_iou:
            print(other_path)


if __name__ == '__main__':
    filepath = sys.argv[1]
    others_root = sys.argv[2]
    others_paths = glob(f'{others_root}/**/*.svg', recursive=True)
    find_duplicates(filepath, others_paths)
