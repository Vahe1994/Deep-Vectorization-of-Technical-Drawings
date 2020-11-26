import numpy as np

from util_files.rendering.cairo import render as original_render
from util_files.metrics.raster_metrics import iou_score
from util_files.data.graphics.graphics import VectorImage
from util_files.data.graphics.units import Pixels


def render(data, dimensions, data_representation='paths', linecaps='butt', linejoin='miter'):
    return original_render(data, dimensions, data_representation=data_representation, linecaps=linecaps,
                           linejoin=linejoin)


def iou_vector_reference(output_vector_image: VectorImage, reference_vector_image: VectorImage, width='mean',
                         binarization='median'):
    output_vector_image = output_vector_image.with_filled_removed()
    reference_vector_image = reference_vector_image.with_filled_removed()

    if (width == 'mean') or isinstance(width, (int, float)):
        if width == 'mean':
            width_to_set = np.array([float(path.width.as_pixels()) for path in reference_vector_image.paths if
                                     path.width is not None]).mean()
        else:
            width_to_set = width
        for path in output_vector_image.paths:
            if path.width is not None:
                path.width = Pixels(width_to_set)
        for path in reference_vector_image.paths:
            if path.width is not None:
                path.width = Pixels(width_to_set)
    # leave as is if width is None

    output_rasterization = output_vector_image.render(render)
    reference_rasterization = reference_vector_image.render(render)

    # undo paddings
    h, w = reference_rasterization.shape
    output_rasterization = output_rasterization[:h, :w]
    return iou_score(reference_rasterization, output_rasterization, binarization=binarization).item()


def iou_raster_reference(output_vector_image: VectorImage, reference_raster_image_np: np.ndarray, width='mean',
                         binarization='median'):
    output_vector_image = output_vector_image.with_filled_removed()

    if (width == 'mean') or isinstance(width, (int, float)):
        if width == 'mean':
            shaded_surface_area = (1 - reference_raster_image_np.astype(np.float32) / 255).sum()
            total_vector_length = sum(
                prim.segment.length() for path in output_vector_image.paths for prim in path if path.visible)
            width_to_set = shaded_surface_area / total_vector_length
        else:
            width_to_set = width
        for path in output_vector_image.paths:
            if path.width is not None:
                path.width = Pixels(width_to_set)
    # leave as is if width is None

    output_rasterization = output_vector_image.render(render)

    if reference_raster_image_np.ndim > 2:
        reference_raster_image_np = reference_raster_image_np[..., 0]

    # undo paddings
    h, w = reference_raster_image_np.shape
    output_rasterization = output_rasterization[:h, :w]
    return iou_score(reference_raster_image_np, output_rasterization, binarization=binarization).item()
