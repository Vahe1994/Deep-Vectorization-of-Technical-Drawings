from math import ceil
from copy import deepcopy

from .utils.raster_utils import image_to_datauri


class RasterEmbedded:
    def __init__(self, image, pos, size, z):
        self.image = image
        self.x, self.y = pos
        self.width, self.height = size
        self.z = z

    def get_svg_code(self):
        datauri = image_to_datauri(self.image)
        return f'\t<image x="{self.x}" y="{self.y}" height="{self.height}" width="{self.width}" xlink:href="{datauri}"/>\n'

    def copy(self):
        return deepcopy(self)

    def crop(self, bbox_in_parent):
        left, right, bottom, top = bbox_in_parent

        # get crops of the box
        left_crop = max(left - self.x, 0)
        bottom_crop = max(bottom - self.y, 0)
        right_crop = max(self.x + self.width - right, 0)
        top_crop = max(self.y + self.height - top, 0)

        # crop data
        h, w = self.image.shape[:2]
        left_crop_data = int(ceil(left_crop / self.width * w))
        right_crop_data = int(ceil(right_crop / self.width * w))
        bottom_crop_data = int(ceil(bottom_crop / self.height * h))
        top_crop_data = int(ceil(top_crop / self.height * h))
        self.image = self.image[bottom_crop_data: h - top_crop_data, left_crop_data: w - right_crop_data]

        # update parameters
        self.x += left_crop
        self.y += bottom_crop
        self.width -= left_crop + right_crop
        self.height -= bottom_crop + top_crop

    def mirror(self, x):
        raise NotImplementedError

    def rotate(self, rotation_deg, origin):
        raise NotImplementedError

    def scale(self, scaling_factor):
        for attr in self.x, self.y, self.width, self.height:
            attr.scale(scaling_factor)

    def translate(self, t):
        self.x += t[0]
        self.y += t[1]

    def translated(self, t):
        r = self.copy()
        r.translate(t)
        return r
