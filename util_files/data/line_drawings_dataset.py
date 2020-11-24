from typing import Dict, List
import random

import numpy as np
from torch.utils.data import Dataset

from util_files.data.graphics_primitives import GraphicsPrimitive, repr_len_by_type
from util_files.rendering.cairo import render
from util_files.color_utils import img_8bit_to_float, rgb_to_gray, gray_float_to_8bit


class LineDrawingsDataset(Dataset):
    def __init__(self, *, patch_size, raster_transform=None, vector_transform=None,
                 primitive_types=None, min_primitives=0, max_primitives=10,
                 normalize_image=False,
                 sample_primitives_randomly=True, sort_primitives=True, pad_empty_primitives=True, **kwargs):
        r"""Hi!

        :param primitive_types: tuple of PrimitiveType's
        """
        self.patch_size = patch_size
        self.raster_transform = raster_transform
        self.vector_transform = vector_transform

        primitive_types = tuple(primitive_types)
        self.primitive_types = primitive_types

        if isinstance(min_primitives, int):
            self.min_primitives = {primitive_type: min_primitives for primitive_type in primitive_types}
        elif isinstance(min_primitives, dict):
            self.min_primitives = min_primitives
        else:
            raise TypeError('min_primitives should be int or dict with the value for each of the primitive_types')

        if isinstance(max_primitives, int):
            self.max_primitives = {primitive_type: max_primitives for primitive_type in primitive_types}
        elif isinstance(max_primitives, dict):
            self.max_primitives = max_primitives
        else:
            raise TypeError('max_primitives should be int or dict with the value for each of the primitive_types')

        self.normalize_image = normalize_image
        self.sample_primitives_randomly = sample_primitives_randomly
        self.sort_primitives = sort_primitives
        self.pad_empty_primitives = pad_empty_primitives


    def _get_vector_item(self, idx: None) -> Dict[str, np.ndarray]: ## ~~List[GraphicsPrimitive]]:~~
        r"""Yields a collection of primitives from a dataset.

        :param idx: unused
        :returns primitives: a dictionary filled with a list of ~~primitives~~ Vahe representations of primitives for each of self.primitive_types
        """
        raise NotImplementedError


    def _render(self, primitive_sets):
        return render(primitive_sets, self.patch_size, data_representation='vahe')


    def __getitem__(self, idx):
        # generate vector graphics
        vector = self._get_vector_item(idx)

        # TODO remove this code when all child classes check this inside
        for primitive_type in self.primitive_types:
            assert len(vector[primitive_type]) <= self.max_primitives[primitive_type], 'Number of {} is {}, max is {}'.format(primitive_type, len(vector[primitive_type]), self.max_primitives[primitive_type])
            #primitives = vector[primitive_type]
            #max_primitives = self.max_primitives[primitive_type]
            #num_primitives = len(primitives)
            #if num_primitives > max_primitives:
            #    if self.sample_primitives_randomly:
            #        primitive_ids = random.sample(range(num_primitives), max_primitives)
            #        vector[primitive_type] = [primitives[idx] for idx in primitive_ids]
            #    else:
            #        vector[primitive_type] = primitives[:max_primitives]

        # possibly apply vector augmentations
        if self.vector_transform:
            vector = self.vector_transform(vector)

        # generate raster image and possibly apply raster augmentations
        raster = self._render(vector)
        raster = img_8bit_to_float(rgb_to_gray(raster))
        if self.raster_transform:
            raster = self.raster_transform(raster)
        if not self.normalize_image:
            raster = gray_float_to_8bit(raster)

        # form a raster - vector pair
        vector_params = {}
        for primitive_type in self.primitive_types:
            primitives_repr = vector[primitive_type]
            max_primitives = self.max_primitives[primitive_type]

            if primitives_repr:
                if self.sort_primitives:
                    primitives_repr.sort()

                primitives_repr = np.array(primitives_repr)
                # add indicators of existing primitives
                num_primitives, num_params = primitives_repr.shape
                primitives_repr = np.concatenate((primitives_repr, np.ones((num_primitives, 1))), axis=1)

                # add indicators of non-existing primitives to fill
                # the array up to required size
                if self.pad_empty_primitives:
                    padding = np.zeros((max_primitives - num_primitives, num_params + 1))
                    primitives_repr = np.concatenate((primitives_repr, padding), axis=0)
            else:
                num_primitives, num_params = 0, repr_len_by_type[primitive_type]
                if self.pad_empty_primitives:
                    padding = np.zeros((max_primitives - num_primitives, num_params + 1))
                    primitives_repr = padding


            vector_params[primitive_type] = primitives_repr

        sample = {'raster': raster, 'vector': vector_params}
        return sample
