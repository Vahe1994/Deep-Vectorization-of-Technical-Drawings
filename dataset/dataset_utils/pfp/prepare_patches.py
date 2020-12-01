import sys

import numpy as np

sys.path.append('/code')
from util_files.data.vectordata.utils.prepatching import make_patches

if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    num_workers = -1
    patch_w, patch_h = 64, 64
    widths = list(range(1, 7 + 1))
    rotations = np.linspace(0, 360, 360 // 5, endpoint=False)

    translations = np.asarray(np.meshgrid(
        np.linspace(0, patch_w, 4 * 3, endpoint=False),
        np.linspace(0, patch_h, 4 * 3, endpoint=False)
    )).transpose(1, 2, 0).reshape(-1, 2)

    augmentations_n_per_image = 4  # !!! crucial to have minimal amount or will take forever

    make_patches(data_root=input_dir, patches_root=output_dir, patch_size=(patch_w, patch_h), remove_filled=True,
                 distinguishability_threshold=1.5, num_workers=num_workers, min_widths=widths, mirror=True,
                 rotations=rotations, translations=translations, random_samples=augmentations_n_per_image,
                 leave_width_percentile=5)
