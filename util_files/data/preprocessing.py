import numpy as np

import util_files.data.graphics_primitives as graphics_primitives


class PreprocessedBase:
    def __init__(self, *, patch_size, **kwargs):
        patch_width, patch_height = patch_size
        assert patch_width == patch_height
        self.patch_width = patch_width

        self._xs = np.arange(1, patch_width + 1, dtype=np.float32)[None].repeat(patch_height, 0) / patch_width
        self._ys = np.arange(1, patch_height + 1, dtype=np.float32)[..., None].repeat(patch_width, 1) / patch_height

    def __getitem__(self, idx):
        r''' Child should do
            return self.preprocess_sample(DatasetClass.__getitem__(self, idx))
        '''
        raise NotImplementedError

    def preprocess_image(self, image):
        image = 1 - image  # 0 -- background
        mask = (image > 0).astype(np.float32)
        return np.stack([image, self._xs * mask, self._ys * mask], axis=-3)

    def preprocess_primitives(self, primitives):
        primitives[..., :-1] /= self.patch_width
        return primitives

    def preprocess_sample(self, sample):
        # TODO: this method is nongeneral -- either generalize or exclude from the general pipeline
        return self.preprocess_image(sample['raster'].astype(np.float32)), \
               self.preprocess_primitives(
                   sample['vector'][graphics_primitives.PrimitiveType.PT_LINE].astype(np.float32))

    @property
    def target_shape(self):
        return [self.max_primitives[graphics_primitives.PrimitiveType.PT_LINE],
                graphics_primitives.repr_len_by_type[graphics_primitives.PrimitiveType.PT_LINE] + 1]


class PreprocessedPacked(PreprocessedBase):
    def preprocess_sample(self, sample):
        raster = self.preprocess_image(sample['raster'].astype(np.float32))
        vector = (self.preprocess_primitives(sample['vector'][prim_type].astype(np.float32))
                  for prim_type in self.primitive_types)
        vector = list(map(self.pack_primitives, vector))
        vector = np.concatenate(vector)
        return raster, vector

    def pack_primitives(self, primitives):
        batch_dims = primitives.ndim - 1
        parameters_n = primitives.shape[-1]
        pad = [[0, 0]] * batch_dims + [[self.max_parameters_per_primitive - parameters_n, 0]]
        return np.pad(primitives, pad, mode='constant')

    @property
    def target_shape(self):
        return [sum(self.max_primitives.values()), self.max_parameters_per_primitive]

    @property
    def max_parameters_per_primitive(self):
        return max(graphics_primitives.repr_len_by_type[prim_type] for prim_type in self.primitive_types) + 1
