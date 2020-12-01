from copy import deepcopy
from glob import glob
import os

import numpy as np

from util_files.data.graphics.graphics import VectorImage


def make_patches(data_root, patches_root, patch_size, outline_filled=None, remove_filled=False, min_widths=('def',),
                 mirror=True, rotations=(0,), translations=((0, 0),), distinguishability_threshold=.5, num_workers=0,
                 random_samples=None, leave_width_percentile=None):
    if num_workers != 0:
        from pathos.multiprocessing import cpu_count, ProcessingPool
        from pathos.threading import ThreadPool
        if num_workers == -1:
            optimal_workers = cpu_count() - 1
            workers_pool = ProcessingPool(optimal_workers)
        else:
            workers_pool = ProcessingPool(num_workers)
        print(f'Workers pool: {workers_pool}')

        savers_pool = ThreadPool(1)
        saving_patches_in_bg = savers_pool.amap(lambda a: None, [])
    else:
        workers_pool = 0

    path = lambda basename, origin, width='def', ori='def', rot=0, t=(0, 0): os.path.join(patches_root, basename,
                                                                                          '{}x{}'.format(*patch_size),
                                                                                          'width_{}'.format(width),
                                                                                          'orientation_{}'.format(ori),
                                                                                          'rotated_deg_{}'.format(rot),
                                                                                          'translated_{}_{}'.format(*t),
                                                                                          '{}_{}.svg'.format(*origin))

    orientations = ['def']
    if mirror:
        orientations.append('mir')

    if random_samples is not None:
        min_widths_all = deepcopy(min_widths)
        orientations_all = deepcopy(orientations)
        rotations_all = deepcopy(rotations)
        translations_all = deepcopy(translations)

    source_images = glob(os.path.join(data_root, '**', '*.svg'), recursive=True)
    for file in source_images:
        print('Processing file {}'.format(file))
        basename = file[len(data_root) + 1:-4]  # split data_root and extension

        vector_image = VectorImage.from_svg(file)
        if remove_filled:
            vector_image.remove_filled()
        if outline_filled is not None:
            vector_image.leave_only_contours(outline_filled)
        if leave_width_percentile is not None:
            vector_image.leave_width_percentile(leave_width_percentile)

        if random_samples is not None:
            min_widths = np.random.choice(min_widths_all, size=min(random_samples, len(min_widths_all)), replace=False)
            orientations = np.random.choice(orientations_all, size=min(random_samples, len(orientations_all)),
                                            replace=False)
            rotations = np.random.choice(rotations_all, size=min(random_samples, len(rotations_all)), replace=False)
            translations = translations_all[
                np.random.choice(len(translations_all), size=min(random_samples, len(translations_all)), replace=False)]

        for width in min_widths:
            print('\twidth {}'.format(width))
            if width == 'def':
                vector_image_scaled = vector_image
            else:
                vector_image_scaled = vector_image.copy()
                vector_image_scaled.scale_to_width('min', width)
            for orientation in orientations:
                print('\t\torientation {}'.format(orientation))
                if orientation == 'def':
                    vector_image_reoriented = vector_image_scaled
                else:
                    vector_image_reoriented = vector_image_scaled.mirrored()
                for rotation in rotations:
                    print('\t\t\trotation {}'.format(rotation))
                    vector_image_rotated = vector_image_reoriented.rotated(rotation, adjust_view=True)
                    for translation in translations:
                        print('\t\t\t\ttranslation {}'.format(translation))
                        vector_image_translated = vector_image_rotated.translated(translation, adjust_view=True)

                        vector_patches = vector_image_translated.split_to_patches(patch_size, workers=workers_pool)
                        if num_workers != 0:
                            print('\t\t\t\t\twaiting for previous batch to be saved')
                            saving_patches_in_bg.get()

                        def simplify_and_save(vector_patch, basename=basename, width=width, orientation=orientation,
                                              rotation=rotation, translation=translation):
                            vector_patch.simplify_segments(distinguishability_threshold=distinguishability_threshold)
                            if len(vector_patch.paths) == 0:
                                return
                            save_path = path(basename,
                                             (int(vector_patch.x.as_pixels()), int(vector_patch.y.as_pixels())), width,
                                             orientation, rotation, translation)
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            vector_patch.save(save_path)

                        if num_workers == 0:
                            print('\t\t\t\t\tsaving patches')
                            for vector_path in vector_patches.reshape(-1):
                                simplify_and_save(vector_path)
                        else:
                            print('\t\t\t\t\tsaving patches')
                            saving_patches_in_bg = savers_pool.amap(simplify_and_save, vector_patches.reshape(-1))

    if num_workers != 0:
        workers_pool.close()
        workers_pool.join()
        workers_pool.clear()

        savers_pool.close()
        savers_pool.join()
        savers_pool.clear()
