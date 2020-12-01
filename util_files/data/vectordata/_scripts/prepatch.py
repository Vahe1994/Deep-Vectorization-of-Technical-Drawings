import sys

import numpy as np

# sys.path.append('/home/oyvoinov/work/3ddl/vectorization/FloorplanVectorization')
sys.path.append('/code')
from util_files.data.vectordata.utils.prepatching import make_patches


def prepatch_sesyd(data_root, patches_root, only_walls=False, num_workers=-1):
    patch_w, patch_h = 64, 64

    widths = list(range(1, 7 + 1))

    rotations = [0, 90, 180, 270]
    max_rot_deviation = 10
    rotations += [base_rot + dev_rot * dev_sign for dev_rot in range(1, max_rot_deviation + 1, 1) for base_rot in
                  (0, 90, 180, 270) for dev_sign in (1, -1)]

    if False:
        translations = list(((tx, ty) for ty in range(patch_h) for tx in range(patch_w)))
    else:
        translations = list(((tx, ty) for ty in (0, int(patch_h // 2)) for tx in (0, int(patch_w // 2))))

    outline_filled = '7px' if only_walls else None

    make_patches(data_root=data_root, patches_root=patches_root, patch_size=(patch_w, patch_h),
                 outline_filled=outline_filled, distinguishability_threshold=1.5, num_workers=num_workers,
                 min_widths=widths, mirror=True, rotations=rotations, translations=translations)


def prepatch_pfp(data_root, patches_root, num_workers=-1):
    patch_w, patch_h = 64, 64

    widths = list(range(1, 3 + 1))

    rotations = np.linspace(0, 360, 360 // 5, endpoint=False)

    translations = np.asarray(np.meshgrid(
        np.linspace(0, patch_w, 4 * 3, endpoint=False),
        np.linspace(0, patch_h, 4 * 3, endpoint=False)
    )).transpose(1, 2, 0).reshape(-1, 2)

    random_samples = 1  # !!! crucial or will take forever

    make_patches(data_root=data_root, patches_root=patches_root, patch_size=(patch_w, patch_h), remove_filled=True,
                 distinguishability_threshold=1.5, num_workers=num_workers, min_widths=widths, mirror=True,
                 rotations=rotations, translations=translations, random_samples=random_samples,
                 leave_width_percentile=5)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        input_dir = '/data/svg_datasets/processed_data/precision-floorplan'
    else:
        input_dir = sys.argv[1]
    # prepatch_sesyd('/home/ovoinov/work/3ddl/vectorization/dataset/processed_data/sesyd',
    #                '/home/ovoinov/work/3ddl/vectorization/dataset/prepatched_data/sesyd', num_workers=-1)
    # prepatch_sesyd('/home/oyvoinov/datasets/svg_datasets/additional/sesyd_walls',
    #                '/home/oyvoinov/datasets/svg_datasets/patched/sesyd_walls', num_workers=0)
    prepatch_pfp(input_dir, '/data/svg_datasets/patched/precision-floorplan', num_workers=-1)
