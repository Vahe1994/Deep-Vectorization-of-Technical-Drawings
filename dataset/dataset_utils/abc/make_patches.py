from contextlib import contextmanager
import os
import signal
import sys

import numpy as np

# sys.path.append('/code')
sys.path.append('/home/vage/PycharmProjects/Deep-Vectorization-of-Technical-Drawings/')
from util_files.data.graphics.graphics import VectorImage, Path
from util_files.simplification.join_qb import join_quad_beziers
from util_files.rendering.cairo import render, PT_LINE, PT_QBEZIER
from util_files.simplification.detect_overlaps import has_overlaps


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def prepare_patch(patch, patch_size, tiny_segments_threshold=1.5, quad_beziers_fit_tol=.1, quad_beziers_w_tol=.1, max_relative_overlap=.5):
    # 1. Remove short segments
    patch.remove_tiny_segments(tiny_segments_threshold)
    if len(patch.paths) == 0:
        return None

    # 2. Simplify curves
    lines, curves = patch.vahe_representation()
    lines = np.asarray(lines)
    if len(curves) > 0:
        curves = join_quad_beziers(curves, fit_tol=quad_beziers_fit_tol, w_tol=quad_beziers_w_tol).numpy()
    curves = np.asarray(curves)

    # 3. Skip the patch if it has overlays
    def render_primitives(primitives):
        return render(primitives, patch_size, data_representation='vahe')

    if has_overlaps({PT_LINE: lines, PT_QBEZIER: curves}, render_primitives, max_relative_overlap=max_relative_overlap):
        return None

    # 4. Convert lines to curves
    curves = curves.tolist() + [line_to_curve(line) for line in lines]
    patch.paths = ([Path.from_primitive(PT_QBEZIER, prim) for prim in curves])
    return patch


def line_to_curve(line):
    p1 = line[:2]
    p3 = line[2:4]
    p2 = (p1 + p3) / 2
    width = line[4]
    return np.array([*p1, *p2, *p3, width])


if __name__ == '__main__':
    '''
    Augmentation is performed on whole image than it splits into patches with something on it.
    
    patch_with: with of the patches 
    path height: height of the patches
    mirroring: is augmentation with mirroring 
    rotation_min, rotation_max: range of the rotations
    augmentations_n: number of augmentation
    '''
    filename = sys.argv[1]
    patches_dir = sys.argv[2]

    patch_width = 64
    patch_height = 64
    distinguishability_threshold = 1.5

    width_min = 1
    width_max = 7
    mirroring = True
    rotation_min = 0
    rotation_max = 360
    translation_x_min = 0
    translation_x_max = patch_width
    translation_y_min = 0
    translation_y_max = patch_height

    augmentations_n = 4

    vector_image = VectorImage.from_svg(filename)

    for augmentation_i in range(augmentations_n):
        width = np.random.random() * (width_max - width_min) + width_min
        mirror = bool(np.random.randint(0, 2)) & mirroring
        rotation = np.random.random() * (rotation_max - rotation_min) + rotation_min
        translation = (np.random.rand(2) *
                       [translation_x_max - translation_x_min, translation_y_max - translation_y_min] +
                       [translation_x_min, translation_y_min])

        augmented_image = vector_image.copy()
        augmented_image.scale_to_width('min', width)
        if mirror:
            augmented_image.mirror()
        augmented_image.rotate(rotation)
        augmented_image.translate(translation, adjust_view=True)

        patches = augmented_image.split_to_patches((patch_width, patch_height)).reshape(-1)
        print(len(patches))
        basename = os.path.basename(filename)[:-4]
        orientation = {False: 'o', True: 'm'}[mirror]
        for patch_i, patch in enumerate(patches):
            try:
                with time_limit(2):
                    patch = prepare_patch(patch, (patch_width, patch_height))
            except TimeoutException:
                print(f'Time exceeded for patch {patch_i} in {basename}')
                continue
            else:
                if patch is None:
                    continue
                save_path = (f'{patches_dir}/{basename}/{patch_width}x{patch_height}/'
                             f'width_{width:.2f}_ori_{orientation}_rot_{rotation:.2f}_'
                             f'tr_{translation[0]:.2f}_{translation[1]:.2f}_'
                             f'{int(patch.x.as_pixels())}_{int(patch.y.as_pixels())}.svg')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                patch.save(save_path)
