from copy import deepcopy
import itertools
from sys import maxsize
import random
import re

import numpy as np
import svgpathtools

from .utils import common as common_utils, parse
from .primitives import Bezier, Line
from .path import Path
from .raster_embedded import RasterEmbedded
from . import units

# import pathos

'''Arcs are ignored atm.'''


class VectorImage:
    def __init__(self, paths, view_size, origin=None, size=None, view_origin=None):
        self.paths = list(paths)
        self.rasters_embedded = []

        if view_origin is not None:
            self.view_x, self.view_y = view_origin
        else:

            self.view_x, self.view_y = [units.Pixels(0), units.Pixels(0)]
        self.view_width, self.view_height = view_size
        self.view_width = self.view_width.copy()  # hack
        self.view_height = self.view_height.copy()  # hack

        if origin is not None:
            self.x, self.y = origin
        if size is not None:
            self.width, self.height = size

    @classmethod
    def from_svg(cls, file):
        # read svg
        paths, path_attribute_dicts, svg_attributes = svgpathtools.svg2paths2(file)

        # get canvas sizes
        x = y = view_x = view_y = width = height = view_width = view_height = None

        if 'x' in svg_attributes or 'y' in svg_attributes:
            origin = [units.Pixels(0), units.Pixels(0)]
            if 'x' in svg_attributes:
                origin[0] = units.fromrepr(svg_attributes['x'], units.Pixels)
            if 'y' in svg_attributes:
                origin[1] = units.fromrepr(svg_attributes['y'], units.Pixels)
        else:
            origin = None

        if 'width' in svg_attributes or 'height' in svg_attributes:
            size = [units.Pixels(0), units.Pixels(0)]
            if 'width' in svg_attributes:
                size[0] = units.fromrepr(svg_attributes['width'], units.Pixels)
            if 'height' in svg_attributes:
                size[1] = units.fromrepr(svg_attributes['height'], units.Pixels)
        else:
            size = None

        if 'viewBox' in svg_attributes:
            view_x, view_y, view_width, view_height = (units.fromrepr(coord, units.Pixels) for coord in
                                                       re.split(' |,', svg_attributes['viewBox']))
            view_origin = (view_x, view_y)
            view_size = (view_width, view_height)
        else:
            view_origin = [units.Pixels(0), units.Pixels(0)]
            view_size = size.copy()

        # convert attributes
        path_attribute_dicts = ({**attribute_dict, **parse.split_path_style_to_attributes(
            attribute_dict['style'])} if 'style' in attribute_dict else attribute_dict for attribute_dict in
                                path_attribute_dicts)

        # create image
        vector_image = cls(
            filter(None,
                   (Path.create_visible(path, attributes) for (path, attributes) in zip(paths, path_attribute_dicts))),
            origin=origin, size=size, view_origin=view_origin, view_size=view_size)

        return vector_image

    @classmethod
    def from_grid(cls, images, stack_rows_first=True, margin_width=None, margin_color=None):
        if stack_rows_first:
            for row in images:
                row_img = row[0].copy()
                for other_img in row[1:]:
                    top = 0
                    bottom = row_img.height
                    left = row_img.width
                    if margin_width is not None:
                        left = left + margin_width / 2
                        if margin_color is not None:
                            margin = Path.make_line([left.as_pixels().value, top],
                                                    [left.as_pixels().value, bottom.as_pixels().value], margin_width,
                                                    color=margin_color)
                            row_img.paths.append(margin)
                        left = left + margin_width / 2
                        row_img.width = row_img.width + margin_width
                        row_img.view_width = row_img.view_width + margin_width
                    row_img.paths = row_img.paths + [path.translated([left.as_pixels().value, 0]) for path in
                                                     other_img.paths]
                    row_img.rasters_embedded = row_img.rasters_embedded + [
                        raster.translated([left.as_pixels().value, 0]) for raster in other_img.rasters_embedded]
                    row_img.width = row_img.width + other_img.width
                    row_img.view_width = row_img.view_width + other_img.view_width
                try:
                    full_img
                except UnboundLocalError:
                    full_img = row_img
                    continue
                left = 0
                right = full_img.width
                bottom = full_img.height
                if margin_width is not None:
                    bottom = bottom + margin_width / 2
                    if margin_color is not None:
                        margin = Path.make_line([left, bottom.as_pixels().value],
                                                [right.as_pixels().value, bottom.as_pixels().value], margin_width,
                                                color=margin_color)
                        full_img.paths.append(margin)
                    bottom = bottom + margin_width / 2
                    full_img.height = full_img.height + margin_width
                    full_img.view_height = full_img.view_height + margin_width
                full_img.paths = full_img.paths + [path.translated([0, bottom.as_pixels().value]) for path in
                                                   row_img.paths]
                full_img.rasters_embedded = full_img.rasters_embedded + [
                    raster.translated([0, bottom.as_pixels().value]) for raster in row_img.rasters_embedded]
                full_img.height = full_img.height + row_img.height
                full_img.view_height = full_img.view_height + row_img.view_height
        return full_img

    def add_frame(self, width, color='black', padding=None):
        if padding is None:
            padding = units.Pixels(0)
        if isinstance(padding, (list, tuple)):
            left, right, top, bottom = padding
            left = -left
            top = -top
            right = right + self.width
            bottom = bottom + self.height
        else:
            left, right = -padding, self.width + padding
            top, bottom = -padding, self.height + padding

        frame = Path.make_rectangle(left.as_pixels().value, top.as_pixels().value,
                                    bottom.as_pixels().value, right.as_pixels().value, width=width, color=color)
        self.paths.append(frame)
        self.adjust_view()

    def add_raster(self, raster_image, pos=None, size=None, z=1):
        if pos is None:
            pos = [self.view_x.copy(), self.view_y.copy()]
        if size is None:
            size = [self.view_width.copy(), self.view_height.copy()]
        self.rasters_embedded.append(RasterEmbedded(raster_image, pos=pos, size=size, z=z))

    def adjust_view(self, margin=.01):
        self.translate((0, 0), adjust_view=True, margin=margin)

    def adjust_viewbox(self):
        if not hasattr(self, 'width'):
            return
        width = self.width.copy()
        height = self.height.copy()
        scale_width = width / self.view_width
        scale_height = height / self.view_height
        self.scale(min(scale_width, scale_height), only_coordinates=True)
        self.width = width
        self.height = height

    def copy(self):
        return deepcopy(self)

    def crop(self, bbox, exact=False):
        [path.crop(bbox) for path in self.paths]
        [raster.crop(bbox) for raster in self.rasters_embedded]
        self.paths = list(filter(lambda path: len(path) > 0, self.paths))
        if exact:
            frame = Path.make_rectangle(bbox[0], bbox[2], bbox[3], bbox[1], width=units.Pixels(0))
            self.paths.append(frame)
            self.adjust_view(margin=0)
            del self.paths[-1]
        else:
            self.adjust_view(margin=0)

    def has_overlapping_widths(self, intersection_tol=1e-1):
        for i in range(len(self.paths)):
            path1 = self.paths[i]
            for j in range(i + 1, len(self.paths)):
                path2 = self.paths[j]
                minimal_distance = (path1.width + path2.width).as_pixels().value / 2
                for prim1 in path1:
                    for prim2 in path2:
                        distance = prim1.distance_to(prim2)
                        if distance < intersection_tol:
                            continue
                        if distance < minimal_distance:
                            return True
        return False

    def leave_only_contours(self, default_width):
        [path.remove_fill(default_width) for path in self.paths]

    def leave_width_percentile(self, percentile):
        widths = np.fromiter((float(path.width.as_pixels()) for path in self.paths if path.width is not None),
                             dtype=np.float)
        min_width = np.percentile(widths, percentile)
        self.paths = list(
            filter(lambda path: (path.width is None) or (path.width.as_pixels() >= min_width), self.paths))

    def mirror(self, *args, **kwargs):
        x = self.view_x.value + self.view_width.value / 2
        [path.mirror(x) for path in self.paths]
        [raster.mirror(x) for raster in self.rasters_embedded]

    def mirrored(self, *args, **kwargs):
        newim = self.copy()
        newim.mirror(*args, **kwargs)
        return newim

    def pad(self, left=None, right=None, top=None, bottom=None):
        if bottom is not None:
            self.view_height = self.view_height + bottom
            if hasattr(self, 'height'):
                self.height = self.view_height.copy()  # hack
        if right is not None:
            self.view_width = self.view_width + right
            if hasattr(self, 'width'):
                self.width = self.view_width.copy()  # hack

    def put_rasters(self, svgfilename):
        with open(svgfilename, 'r+') as file:
            lines = file.readlines()
            underlays = list(filter(lambda r: r.z == 1, self.rasters_embedded))
            overlays = list(filter(lambda r: r.z == -1, self.rasters_embedded))
            if len(underlays) > 0:
                for i in range(len(lines)):
                    if '<path' in lines[i]:
                        break
                lines[i:i] = [raster.get_svg_code() for raster in underlays]
            if len(overlays) > 0:
                lines[-1:-1] = [raster.get_svg_code() for raster in overlays]
            file.seek(0)
            file.writelines(lines)

    def remove_duplicates(self):
        unique_paths = []
        for path in self.paths:
            if path not in unique_paths:
                unique_paths.append(path)

        self.paths = []
        for path in unique_paths:
            unique_segments_of_this_path = []
            for segment in path:
                segment_is_unique = True
                for other_path in self.paths:
                    if path.has_same_attributes(other_path) and (segment in other_path):
                        segment_is_unique = False
                        break
                if segment_is_unique:
                    unique_segments_of_this_path.append(segment)
            if len(unique_segments_of_this_path) > 0:
                path.segments = unique_segments_of_this_path
                self.paths.append(path)

    def remove_filled(self):
        self.paths = [path for path in self.paths if not path.fill]

    def with_filled_removed(self):
        newim = self.copy()
        newim.remove_filled()
        return newim

    def remove_tiny_segments(self, threshold):
        self.paths = list(filter(lambda path: path.nonempty,
                                 (path.with_removed_tiny_segments(threshold) for path in self.paths)))

    def render(self, renderer):
        if (self.view_x == 0) and (self.view_y == 0):
            paths = self.paths
        else:
            paths = [path.translated((-self.view_x.as_pixels().value, -self.view_y.as_pixels().value)) for path in
                     self.paths]
        return renderer(paths, (round(self.view_width.as_pixels()), round(self.view_height.as_pixels())))

    def rotate(self, rotation_deg, origin=None, adjust_view=False, margin=.01):
        if origin is None:
            origin = self.view_x.value + self.view_width.value / 2, self.view_y.value + self.view_height.value / 2
        [path.rotate(rotation_deg, origin) for path in self.paths]
        [raster.rotate(rotation_deg, origin) for raster in self.rasters_embedded]

        if adjust_view:
            self.adjust_view(margin=margin)

    def rotated(self, *args, **kwargs):
        newim = self.copy()
        newim.rotate(*args, **kwargs)
        return newim

    def save(self, file, **kwargs):
        if file[-3:].lower() == 'svg':
            return self._save2svg(file, **kwargs)
        else:
            raise NotImplementedError('Unknown file format {}'.format(file))

    def _save2svg(self, file, **kwargs):
        svg_attributes = {}
        if hasattr(self, 'x'):
            svg_attributes['x'] = str(self.x)
            svg_attributes['y'] = str(self.y)
        if hasattr(self, 'width'):
            svg_attributes['width'] = str(self.width)
            svg_attributes['height'] = str(self.height)
        else:
            svg_attributes['width'] = str(self.view_width)
            svg_attributes['height'] = str(self.view_height)
        # if typeerror TypeError: is not a valid value for attribute 'viewBox' at svg-element <svg>.Uncomment 1st one
        # svg_attributes['viewBox'] = '{} {} {} {}'.format(self.view_x, self.view_y, self.view_width, self.view_height)
        svg_attributes['viewBox'] = '{} {} {} {}'.format(int(self.view_x), int(self.view_y), int(self.view_width),
                                                         int(self.view_height))

        svg_attributes['size'] =(svg_attributes['width'], svg_attributes['height'])
        if 'nodes' in kwargs:
            nodes = kwargs['nodes']
        else:
            nodes = None
        if 'node_colors' in kwargs:
            node_colors = kwargs['node_colors']
        else:
            node_colors = None
        if 'node_radii' in kwargs:
            node_radii = kwargs['node_radii']
        else:
            node_radii = None

        paths, attributes = zip(*(path.to_svgpathtools() for path in self.paths))
        print(svg_attributes)
        ret = svgpathtools.wsvg(paths, filename=file, attributes=attributes, svg_attributes=svg_attributes,
                                nodes=nodes, node_colors=node_colors, node_radii=node_radii)
        if len(self.rasters_embedded) > 0:
            self.put_rasters(file)
        return ret

    def scale(self, scale=None, width=None, height=None, only_coordinates=False):
        if (int(scale is not None) + int(width is not None) + int(height is not None)) != 1:
            raise ValueError('Specify only one of scale, width, height')
        if width is not None:
            scale = width.as_pixels().value / self.view_width.as_pixels().value
        if height is not None:
            scale = height.as_pixels().value / self.view_height.as_pixels().value
        self.view_x *= scale
        self.view_y *= scale
        self.view_width *= scale
        self.view_height *= scale
        if hasattr(self, 'width'):
            self.width *= scale
            self.height *= scale

        [path.scale(scale, only_coordinates=only_coordinates) for path in self.paths]
        [raster.scale(scale) for raster in self.rasters_embedded]

    def scale_to_width(self, mode, new_value=None):
        widths = (path.width for path in self.paths if path.width is not None)

        if mode == 'max':
            max_width = max(widths)
            scale = new_value / max_width
        elif mode == 'min':
            min_width = min(widths)
            scale = new_value / min_width
        self.scale(scale)

    def simplify_segments(self, distinguishability_threshold):
        self.paths = list(filter(lambda path: path.nonempty,
                                 (path.with_simplified_segments(distinguishability_threshold) for path in self.paths)))

    def split_to_patches(self, patch_size, workers=0, paths_per_worker=10):
        origin = (float(self.view_x), float(self.view_y))

        patch_w, patch_h = patch_size
        patches_row_n = int(np.ceil(self.view_height.as_pixels() / patch_h))
        patches_col_n = int(np.ceil(self.view_width.as_pixels() / patch_w))
        patches_n = (patches_row_n, patches_col_n)

        vector_patch_origin_xs = np.array([patch_w * j for j in range(patches_col_n)])
        vector_patch_origin_ys = np.array([patch_h * i for i in range(patches_row_n)])
        vector_patch_origins = np.stack((vector_patch_origin_xs[None].repeat(patches_row_n, axis=0),
                                         vector_patch_origin_ys[:, None].repeat(patches_col_n, axis=1)), axis=-1)

        patch_size_pixels = units.Pixels(patch_w), units.Pixels(patch_h)
        vector_patches = np.array([[self.__class__([],
                                                   origin=(units.Pixels(coord) for coord in vector_patch_origins[i, j]),
                                                   size=patch_size_pixels, view_size=patch_size_pixels) for j in
                                    range(patches_col_n)] for i in range(patches_row_n)])

        split_path_to_patches = lambda path: path.split_to_patches(origin=origin, patch_size=patch_size,
                                                                   patches_n=patches_n)

        def distribute_path_in_patches(iS, jS, paths):
            for idx in range(len(iS)):
                i = iS[idx]
                j = jS[idx]
                path_in_patch = paths[idx]
                vector_patches[i, j].paths.append(path_in_patch.translated(-vector_patch_origins[i, j]))

        if isinstance(workers, int) and workers == 0:
            for path in self.paths:
                distribute_path_in_patches(*split_path_to_patches(path))
        else:
            if isinstance(workers, int):
                from pathos.multiprocessing import cpu_count, ProcessingPool as Pool
                if workers == -1:
                    batches_n = int(np.ceil(len(self.paths) / paths_per_worker))
                    optimal_workers = cpu_count() - 1
                    workers = min(optimal_workers, batches_n)
                workers = Pool(workers)
                close_workers = True
            else:
                close_workers = False

            for splits in workers.uimap(split_path_to_patches, self.paths, chunksize=paths_per_worker):
                distribute_path_in_patches(*splits)

            if close_workers:
                workers.close()
                workers.join()
                workers.clear()

        return vector_patches

    def translate(self, translation_vector, adjust_view=False, margin=.01):
        if adjust_view:
            minx, maxx, miny, maxy = common_utils.bbox(seg for path in self.paths for seg in path.segments)
            minx = min([minx] + [raster.x.as_pixels().value for raster in self.rasters_embedded])
            maxx = max([maxx] + [raster.x.as_pixels().value + raster.width.as_pixels().value for raster in
                                 self.rasters_embedded])
            miny = min([miny] + [raster.y.as_pixels().value for raster in self.rasters_embedded])
            maxy = max([maxy] + [raster.y.as_pixels().value + raster.height.as_pixels().value for raster in
                                 self.rasters_embedded])
            self.view_x.value = 0
            self.view_y.value = 0
            self.view_width.value = maxx - minx + translation_vector[0]
            self.view_height.value = maxy - miny + translation_vector[1]

            margin_w = self.view_width.value * margin
            margin_h = self.view_height.value * margin
            self.view_width.value *= (1 + margin * 2)
            self.view_height.value *= (1 + margin * 2)
            if hasattr(self, 'width'):
                self.width = self.view_width.copy()  # hack
                self.height = self.view_height.copy()  # hack

            translation_vector = translation_vector[0] - minx + margin_w, translation_vector[1] - miny + margin_h
            [path.translate(translation_vector) for path in self.paths]
            [raster.translate(translation_vector) for raster in self.rasters_embedded]
        else:
            self.view_x -= translation_vector[0]
            self.view_y -= translation_vector[1]

    def translated(self, *args, **kwargs):
        newim = self.copy()
        newim.translate(*args, **kwargs)
        return newim

    def vahe_representation(self, max_lines_n=maxsize, max_beziers_n=maxsize, random_sampling=False):
        if random_sampling:
            return self._vahe_representation_random(max_lines_n=max_lines_n, max_beziers_n=max_beziers_n)
        else:
            return self._vahe_representation_first(max_lines_n=max_lines_n, max_beziers_n=max_beziers_n)

    def _vahe_representation_first(self, max_lines_n, max_beziers_n):
        lines = list(itertools.islice((seg.vahe_representation(float(path.width.as_pixels())) \
                                       for path in self.paths if path.width is not None \
                                       for seg in path if isinstance(seg, Line)),
                                      max_lines_n))
        beziers = list(itertools.islice((seg.vahe_representation(float(path.width.as_pixels())) \
                                         for path in self.paths if path.width is not None \
                                         for seg in path if isinstance(seg, Bezier)),
                                        max_beziers_n))
        return lines, beziers

    def _vahe_representation_random(self, max_lines_n, max_beziers_n):
        lines, beziers = self._vahe_representation_first(max_lines_n=maxsize, max_beziers_n=maxsize)
        if len(lines) > max_lines_n:
            primitive_ids = random.sample(range(len(lines)), max_lines_n)
            lines = [lines[idx] for idx in primitive_ids]
        if len(beziers) > max_beziers_n:
            primitive_ids = random.sample(range(len(beziers)), max_beziers_n)
            beziers = [beziers[idx] for idx in primitive_ids]
        return lines, beziers
