import torch

from util_files.data.graphics.graphics import Path, VectorImage
from util_files.data.graphics.units import Pixels
from util_files.data.graphics_primitives import PT_LINE, PT_QBEZIER


def vector_image_from_patches(primitives, patch_offsets, control_points_n, patch_size, image_size,
                              pixel_center_coodinates_are_integer=False, scale=None,
                              min_width=.3, min_confidence=.5, min_length=1.7):
    r"""

    Parameters
    ----------
    primitives : array_like
        of shape [patches_n, primitives_n, parameters_n]
    patch_offsets : array_like
        of shape [patches_n, spatial_dims_n]
    patch_size : tuple
        [width, height]

    Returns
    -------

    """
    primitives = torch.tensor(primitives)
    dtype = primitives.dtype
    patch_offsets = torch.tensor(patch_offsets).type(dtype)

    # 0. Fix bug in offsets
    patch_offsets = torch.stack([patch_offsets[:, 1], patch_offsets[:, 0]], dim=1)

    patches_n, primitives_n, parameters_n = primitives.shape
    spatial_dims_n = patch_offsets.shape[1]

    # 1. Shift primitives w.r.t patch offsets
    parameters_dim_i = 1
    primitive_shifts = torch.cat([patch_offsets] * control_points_n +
                                 [patch_offsets.new_zeros(patches_n, 2)], dim=parameters_dim_i)
    primitive_shifts = primitive_shifts.reshape(patches_n, 1, parameters_n)
    primitives = primitives + primitive_shifts
    del primitive_shifts

    # 2. Shift their bounding boxes
    # FIXME: hardcoded spatial_dims_n
    spatial_dims_dim_i = 1
    width, height = patch_size
    minx = 0
    maxx = width
    miny = 0
    maxy = height
    bounding_boxes = torch.tensor([minx, maxx, miny, maxy], dtype=dtype)
    if pixel_center_coodinates_are_integer:
        bounding_boxes -= .5
    box_shifts = patch_offsets.repeat_interleave(2, dim=spatial_dims_dim_i)
    bounding_boxes = bounding_boxes.reshape(1, -1)
    bounding_boxes = bounding_boxes + box_shifts
    del box_shifts, patch_offsets

    # 3. Get rid of patch dimension
    primitives = primitives.reshape(-1, parameters_n)
    bounding_boxes = (bounding_boxes.reshape(patches_n, 1, -1).expand(patches_n, primitives_n, -1)
                                    .reshape(patches_n * primitives_n, -1))

    # 4. Get rid of primitives with small width, low confidence, and nans in parameters
    #    get rid of confidence parameter
    good_primitives = ((primitives[:, -2] >= min_width) & (primitives[:, -1] >= min_confidence) &
                       torch.isfinite(primitives).all(dim=1))
    primitives = primitives[good_primitives, :-1].contiguous()
    bounding_boxes = bounding_boxes[good_primitives].contiguous()
    del good_primitives, patches_n
    primitives_n = len(primitives)
    parameters_n -= 1

    # 5. Get rid of too short primitives
    length = primitives.new_zeros(primitives_n)
    p1 = primitives[:, :spatial_dims_n]
    for p2_i in range(1, control_points_n):
        p2 = primitives[:, spatial_dims_n * p2_i: spatial_dims_n * (p2_i + 1)]
        length += torch.norm(p2 - p1, dim=1)
        p1 = p2
    del p1, p2

    good_primitives = length >= min_length
    primitives = primitives[good_primitives].contiguous()
    bounding_boxes = bounding_boxes[good_primitives].contiguous()
    del length, good_primitives
    primitives_n = len(primitives)

    # 6. Convert primitives to Paths and assemble VectorImage
    # FIXME: hardcoded spatial_dims_n
    paths = list(filter(None, map(primitive_to_path_and_crop, zip(primitives.numpy(), bounding_boxes.numpy()))))
    del primitives, bounding_boxes

    width = Pixels(image_size[1])
    height = Pixels(image_size[0])
    view_size = width, height

    vector_image = VectorImage(paths, view_size=view_size)
    if scale:
        vector_image.scale(scale)
        width = Pixels(image_size[1])
        height = Pixels(image_size[0])
        vector_image.view_width = width
        vector_image.view_height = height
    return vector_image


def primitive_to_path_and_crop(arg):
    primitive_parameters, bbox = arg
    params_n = len(primitive_parameters)
    if params_n == 5:
        path = Path.from_primitive(PT_LINE, primitive_parameters)
    elif params_n:
        path = Path.from_primitive(PT_QBEZIER, primitive_parameters)
    else:
        raise NotImplementedError(f'Unknown primitive with {params_n} parameters')

    path.crop(bbox)
    if len(path) == 0:
        return None
    else:
        return path
