from glob import glob
import os

from util_files.data.line_drawings_dataset import LineDrawingsDataset
from util_files.data.graphics_primitives import PrimitiveType
from .common import get_random_patch_from_svg
from .common import sample_primitive_representation


class _SVGDataset(LineDrawingsDataset):
    r"""
        Arguments:
            patch_size (tuple): patch size in pixels. The patch center is sampled uniformly.
            margin (scalar): how many patches are left around The patch for augmentation.
            ppi (scalar): pixels per inch. Larger values mean wider lines and less rich patches.
            scale_width (scalar): the original path width is multiplied by this value.
            skip_empty (bool): if ``True`` sample patch until it is not empty. Be aware of immenient infinite loop.

        Augmentation arguments:
            random_width (bool): if ``True`` then the width of each path is sampled from gaussian distribution with mean = 3 sigma = the original width.
            rotate (bool): if ``True`` then the whole image is randomly rotated around the patch center.
            per_path_translation_amplitude (complex): if not zero then each path is translated as a whole by a random translation vector with x component uniformly sampled from ``(-1,1) * amplitude.real`` and y component -- from ``(-1,1) * amplitude.imag``.
            per_path_rotation_amplitude (scalar, degrees): if not zero then each path is rotated around its center as a whole by a random amount of degrees uniformly sampled from ``(-1,1) * amplitude``.
            jitter_translation_amplitude (complex): like :attr:`per_path_translation_amplitude` but each point is translated independently.
            jitter_radius_amplitude (scalar): if not zero then for each arc the radius is scaled by a random factor uniformly sampled from ``1 + (-1,1) * jitter_radius_amplitude``.
            jitter_rotation_amplitude (scalar): if not zero then each arc is rotated around its center by a random amount of degrees uniformly sampled from ``(-1,1) * amplitude``.
            mutually_exclusive_transforms (bool): if ``True`` then for each patch only one of the {per_path_translation, per_path_rotation, jitter} transforms is performed.
            autoscale_transforms (bool): if ``True`` then rescale the translation transform parameters such that the visual effect is the same independently of the value of :attr:`ppi`.
    """

    def __init__(self, *, data_root, patch_size, margin=1, ppi=72, scale_width=1, skip_empty=False,
            random_width=False, rotate=False, per_path_translation_amplitude=0, per_path_rotation_amplitude=0,
            jitter_translation_amplitude=0, jitter_radius_amplitude=0, jitter_rotation_amplitude=0,
            mutually_exclusive_transforms=True, autoscale_transforms=True, **kwargs):
        super().__init__(patch_size=patch_size, **kwargs)

        if not set(self.primitive_types).issubset(set([PrimitiveType.PT_LINE, PrimitiveType.PT_BEZIER])):
            raise NotImplementedError('SVG datasets do not support primitives other than Lines and Bezier curves')

        for primitive_type in PrimitiveType.PT_LINE, PrimitiveType.PT_BEZIER:
            if primitive_type not in self.primitive_types:
                self.max_primitives[primitive_type] = 0

        self.source_images = glob(os.path.join(data_root, '**', '*.svg'), recursive=True)
        self.margin = margin
        self.ppi = ppi
        self.scale_width = scale_width
        self.skip_empty = skip_empty

        self.random_width = random_width
        self.rotate = rotate

        self.per_path_translation_amplitude = per_path_translation_amplitude
        self.per_path_rotation_amplitude = per_path_rotation_amplitude
        self.jitter_translation_amplitude = jitter_translation_amplitude
        self.jitter_radius_amplitude = jitter_radius_amplitude
        self.jitter_rotation_amplitude = jitter_rotation_amplitude

        self.mutually_exclusive_transforms = mutually_exclusive_transforms

        if autoscale_transforms:
            parameter_scaling = self.ppi / self.baseline_ppi
            self.per_path_translation_amplitude *= parameter_scaling
            self.jitter_translation_amplitude *= parameter_scaling

    def __len__(self):
        return len(self.source_images)


class PrecisionFloorplan(_SVGDataset):
    def __init__(self, *, data_root, patch_size, margin=1, ppi=300, scale_width=2, skip_empty=False,
            random_width=True, rotate=True, per_path_translation_amplitude=8+8j, per_path_rotation_amplitude=10,
            jitter_translation_amplitude=8+8j, jitter_radius_amplitude=.1, jitter_rotation_amplitude=10,
            mutually_exclusive_transforms=True, autoscale_transforms=True, **kwargs):
        self.baseline_ppi = 600
        super().__init__(data_root=data_root, patch_size=patch_size, margin=margin, ppi=ppi, scale_width=scale_width, skip_empty=skip_empty,
            random_width=random_width, rotate=rotate, per_path_translation_amplitude=per_path_translation_amplitude, per_path_rotation_amplitude=per_path_rotation_amplitude,
            jitter_translation_amplitude=jitter_translation_amplitude, jitter_radius_amplitude=jitter_radius_amplitude, jitter_rotation_amplitude=jitter_rotation_amplitude,
            mutually_exclusive_transforms=mutually_exclusive_transforms, autoscale_transforms=autoscale_transforms, **kwargs)

    def __len__(self):
        return len(self.source_images)

    def _get_vector_item(self, idx):
        filename = self.source_images[idx]

        while True:
            while True:
                paths, attribute_dicts, _ = get_random_patch_from_svg(
                    filename,
                    patch_size=self.patch_size, margin=self.margin, new_ppi_to_old_ppi=self.ppi/72, scale_width=self.scale_width,
                    random_width=self.random_width,
                    rotate=self.rotate, mutually_exclusive_transforms=self.mutually_exclusive_transforms,
                    per_path_translation_amplitude=self.per_path_translation_amplitude, per_path_rotation_amplitude=self.per_path_rotation_amplitude,
                    jitter_translation_amplitude=self.jitter_translation_amplitude, jitter_radius_amplitude=self.jitter_radius_amplitude,
                    remove_filled=True)
                if not self.skip_empty or len(paths) > 0:
                    break

            lines, arcs, beziers = sample_primitive_representation(paths, attribute_dicts, max_lines_n=self.max_primitives[PrimitiveType.PT_LINE], max_beziers_n=self.max_primitives[PrimitiveType.PT_BEZIER], sample_primitives_randomly=self.sample_primitives_randomly)

            if self.primitive_types == (PrimitiveType.PT_LINE,):
                if self.skip_empty and len(lines) == 0:
                    continue
                else:
                    vector = {PrimitiveType.PT_LINE: lines}
            elif self.primitive_types == (PrimitiveType.PT_BEZIER,):
                if self.skip_empty and len(beziers) == 0:
                    continue
                else:
                    vector = {PrimitiveType.PT_BEZIER: beziers}
            else:
                vector = {PrimitiveType.PT_LINE: lines, PrimitiveType.PT_BEZIER: beziers}
            return vector


class SESYD(_SVGDataset):
    r"""
        Additional arguments:
            outline_filled (bool): if ``True`` outline filled paths.
            outline_style (dict): stroke attributes for filled paths outline. Required if :attr:`outline_filled` is ``True``.
    """
    __doc__ = _SVGDataset.__doc__ + __doc__

    def __init__(self, *, data_root, patch_size, margin=1, ppi=72/5, scale_width=1, skip_empty=False,
            random_width=True, rotate=True, per_path_translation_amplitude=16+16j, per_path_rotation_amplitude=10,
            jitter_translation_amplitude=16+16j, jitter_radius_amplitude=.1, jitter_rotation_amplitude=10,
            mutually_exclusive_transforms=True, autoscale_transforms=True,
            outline_filled=True, outline_style={'stroke':'#000000', 'stroke-width':'6.5'}, **kwargs):
        self.baseline_ppi = 72
        super().__init__(data_root=data_root, patch_size=patch_size, margin=margin, ppi=ppi, scale_width=scale_width, skip_empty=skip_empty,
            random_width=random_width, rotate=rotate, per_path_translation_amplitude=per_path_translation_amplitude, per_path_rotation_amplitude=per_path_rotation_amplitude,
            jitter_translation_amplitude=jitter_translation_amplitude, jitter_radius_amplitude=jitter_radius_amplitude, jitter_rotation_amplitude=jitter_rotation_amplitude,
            mutually_exclusive_transforms=mutually_exclusive_transforms, autoscale_transforms=autoscale_transforms, **kwargs)

        self.outline_filled = outline_filled
        self.outline_style = outline_style

    def __len__(self):
        return len(self.source_images)

    def _get_vector_item(self, idx):
        filename = self.source_images[idx]

        while True:
            while True:
                paths, attribute_dicts, _ = get_random_patch_from_svg(filename,
                    patch_size=self.patch_size, margin=self.margin, new_ppi_to_old_ppi=self.ppi/72, scale_width=self.scale_width,
                    random_width=self.random_width,
                    rotate=self.rotate, mutually_exclusive_transforms=self.mutually_exclusive_transforms,
                    per_path_translation_amplitude=self.per_path_translation_amplitude, per_path_rotation_amplitude=self.per_path_rotation_amplitude,
                    jitter_translation_amplitude=self.jitter_translation_amplitude, jitter_radius_amplitude=self.jitter_radius_amplitude,
                    remove_filled=False, outline_filled=self.outline_filled, outline_style=self.outline_style)
                if not self.skip_empty or len(paths) > 0:
                    break

            lines, arcs, beziers = sample_primitive_representation(paths, attribute_dicts, max_lines_n=self.max_primitives[PrimitiveType.PT_LINE], max_beziers_n=self.max_primitives[PrimitiveType.PT_BEZIER], sample_primitives_randomly=self.sample_primitives_randomly)

            if self.primitive_types == (PrimitiveType.PT_LINE,):
                if self.skip_empty and len(lines) == 0:
                    continue
                else:
                    vector = {PrimitiveType.PT_LINE: lines}
            elif self.primitive_types == (PrimitiveType.PT_BEZIER,):
                if self.skip_empty and len(beziers) == 0:
                    continue
                else:
                    vector = {PrimitiveType.PT_BEZIER: beziers}
            else:
                vector = {PrimitiveType.PT_LINE: lines, PrimitiveType.PT_BEZIER: beziers}
            return vector
