# This is loss function modification suggested in DeepParametricShape CVPR paper https://people.csail.mit.edu/smirnov/deep-parametric-shapes/

import numpy as np
import torch as th

import util_files.color_utils as color_utils
from scipy.ndimage.morphology import binary_dilation


def make_safe(x):
    """Set small entries to one to avoid div by zero."""
    return th.where(x.abs() < 1e-6, th.ones_like(x), x)


def distance_to_curves(source_points, curves):
    """Compute the distance from each source point to each quaratic Bezier curve.

    source_points -- [n_points, 2]
    curves -- [..., 3, 2]
    """
    p0, p1, p2 = th.split(curves, 1, dim=-2)  # [..., 1, 2]

    X = p1 - p0  # [..., 1, 2]
    Y = p2 - p1 - X  # [..., 1, 2]
    Z = p0 - source_points  # [..., n_points, 2]

    a = th.sum(Y * Y, dim=-1)  # [..., 1]
    a = make_safe(a)

    b = 3 * th.sum(X * Y, dim=-1)  # [..., 1]
    c = 2 * th.sum(X * X, dim=-1) + th.sum(Y * Z, dim=-1)  # [..., n_points]
    d = th.sum(X * Z, dim=-1)  # [..., n_points]

    A = b / a
    B = c / a
    C = d / a

    Q = (A ** 2 - 3 * B) / 9
    sqrt_Q = th.sqrt(Q.abs() + 1e-6)
    sqrt_Q = make_safe(sqrt_Q)
    R = (2 * A ** 3 - 9 * A * B + 27 * C) / 54

    theta = th.acos(th.clamp(R / sqrt_Q ** 3, -1 + 1e-6, 1 - 1e-6))
    t1 = -2 * sqrt_Q * th.cos(theta / 3) - A / 3
    t2 = -2 * sqrt_Q * th.cos((theta + 2 * np.pi) / 3) - A / 3
    t3 = -2 * sqrt_Q * th.cos((theta + 4 * np.pi) / 3) - A / 3

    alpha = -R.sign() * (R.abs() + th.sqrt(th.abs(R ** 2 - Q ** 3) + 1e-6)) ** (1 / 3)
    alpha = make_safe(alpha)
    beta = Q / alpha

    t4 = alpha + beta - A / 3
    c = make_safe(c)
    t5 = -d / c

    ts = th.stack([t1, t2, t3, t4, t5], dim=-1)  # [..., n_points, 5]
    ts = th.clamp(ts, 1e-6, 1)

    ts = ts[..., None].pow(ts.new_tensor([0, 1, 2]))  # [..., n_points, 5, 3]

    A = ts.new_tensor([[1., 0, 0],
                       [-2, 2, 0],
                       [1, -2, 1]])
    points = ts @ A @ curves.unsqueeze(-3)  # [..., n_points, 5, 2]

    sizes = [-1] * (points.dim() - 3) + [points.shape[-3], -1, -1]
    endpoints = th.cat([p0, p2], dim=-2).unsqueeze(-3).expand(*sizes)  # [..., n_points, 2, 2]
    points = th.cat([points, endpoints], dim=-2)  # [..., n_points, 7, 2]

    distances, _ = th.min(th.sqrt(th.sum((points - source_points[:, None, :]) ** 2, dim=-1) + 1e-6),
                          dim=-1)  # [..., n_points]

    return distances


def unroll_curves(curves, topology):
    """Unroll curve parameters into loops as defined by the topology.

    curves -- [b, 2*max_n_curves, 2]
    topology -- [n_loops] list of curves per loop (should sum to max_n_curves)
    """
    print(curves.shape)
    b = curves.shape[0]
    curves = curves.view(b, -1, 2)
    print(curves.shape)
    loops = th.split(curves, [2 * n for n in topology], dim=1)

    unrolled_loops = []
    for loop in loops:
        print('loop', loop.shape)
        loop = loop.unfold(1, 3, 2)  ## ошибка
        loop = loop.permute(0, 1, 3, 2)
        loop = loop.view(b, -1, 3, 2)
        unrolled_loops.append(loop)
    return unrolled_loops  # n_loops x [b, n_curves, 3, 2]


def compute_distance_fields(curves, canvas_size):
    """Compute distance fields of size (canvas_size+2)^2. Distances corresponding to unused curves are set to 10.

    curves -- [b, max_n_curves, 3, 2]
    canvas_size
    """
    grid_pts = th.stack(
        th.meshgrid([th.linspace(-1 / (canvas_size - 1), 1 + 1 / (canvas_size - 1), canvas_size + 2)] * 2),
        dim=-1).permute(1, 0, 2).reshape(-1, 2).to(curves)

    distances = distance_to_curves(grid_pts, curves).view(-1, curves.shape[1],
                                                          canvas_size + 2, canvas_size + 2)

    return distances


# def compute_distance_fields(curves, n_loops, topology, canvas_size):
#     """Compute distance fields of size (canvas_size+2)^2. Distances corresponding to unused curves are set to 10.
#
#     curves -- [b, 2*max_n_curves, 2]
#     n_loops -- [b] number of loops per batch example
#     topology -- [n_loops] list of curves per loop (should sum to max_n_curves)
#     canvas_size
#     """
#     grid_pts = th.stack(th.meshgrid([th.linspace(-1/(canvas_size-1), 1+1/(canvas_size-1), canvas_size+2)]*2),
#                         dim=-1).permute(1, 0, 2).reshape(-1, 2).to(curves)
#     # print(grid_pts)
#     loops = unroll_curves(curves, topology)
#     distance_fields = []
#     for i, loop in enumerate(loops):
#         idxs = (n_loops>i).nonzero().squeeze()
#         print('idxs',idxs)
#         n_curves = loop.shape[1]
#         padded_distances = 10*loop.new_ones(loop.shape[0], n_curves, canvas_size+2, canvas_size+2)
#
#         if idxs.numel() > 0:
#             distances = distance_to_curves(grid_pts, loop.index_select(0, idxs)).view(-1, n_curves,
#                                                                                       canvas_size+2, canvas_size+2)
#             # print('distances', distances)
#             padded_distances[idxs] = distances
#
#         distance_fields.append(padded_distances)
#     # print('distance_fields', distance_fields)
#     return th.cat(distance_fields, dim=1)


def compute_alignment_fields(distance_fields):
    """Compute alignment unit vector fields from distance fields."""
    # dx = distance_fields[..., 2:, 1:-1] - distance_fields[..., :-2, 1:-1]
    # dy = distance_fields[..., 1:-1, 2:] - distance_fields[..., 1:-1, :-2]
    # Changed dx and dy to make it in right order
    dx = distance_fields[..., 1:-1, 2:] - distance_fields[..., 1:-1, :-2]
    dy = distance_fields[..., 2:, 1:-1] - distance_fields[..., :-2, 1:-1]
    alignment_fields = th.stack([dx, dy], dim=-1)
    return alignment_fields / th.sqrt(th.sum(alignment_fields ** 2, dim=-1, keepdims=True) + 1e-10)


def compute_occupancy_fields(distance_fields, eps=(2 / 128) ** 2):
    """Compute smooth occupancy fields from distance fields."""
    occupancy_fields = 1 - th.clamp(distance_fields / eps, 0, 1)
    return occupancy_fields ** 2 * (3 - 2 * occupancy_fields)


# Used for Champfer loss

def sample_points_from_curves(curves, n_loops, topology, n_samples_per_curve):
    """Sample points from Bezier curves.

    curves -- [b, 2*max_n_curves, 2]
    n_loops -- [b] number of loops per batch example
    topology -- [n_loops] list of curves per loop (should sum to max_n_curves)
    n_samples_per_curve
    """
    A = curves.new_tensor([[1, 0, 0],
                           [-2, 2, 0],
                           [1, -2, 1]])

    loops = unroll_curves(curves, topology)
    all_points = th.empty(curves.shape[0], 0, 2).to(curves)
    for i, loop in enumerate(loops):
        idxs = (n_loops > i).nonzero().squeeze()
        loop = loop.index_select(0, idxs)  # [?, n_curves, 3, 2]
        n_curves = loop.shape[1]

        ts = th.empty(n_curves, n_samples_per_curve).uniform_(0, 1)
        ts = ts[..., None].pow(ts.new_tensor([0, 1, 2])).to(curves)  # [n_points, 3]

        points = ts @ A @ loop  # [?, n_curves, n_points, 2]
        points = points.view(-1, n_samples_per_curve * n_curves, 2)

        if i > 0:
            pad_idxs = th.randperm(all_points.shape[1])[:n_samples_per_curve * n_curves]
            padded_points = all_points[:, pad_idxs]

            padded_points[idxs] = points
        else:
            padded_points = points
        all_points = th.cat([all_points, padded_points], dim=1)

    return all_points


def compute_chamfer_distance(a, b):
    """Compute Chamfer distance between two point sets.

    a -- [b, n, 2]
    b -- [b, m, 2]
    """
    D = th.sqrt(th.sum((a.unsqueeze(1) - b.unsqueeze(2)) ** 2, dim=-1) + 1e-6)  # [b, m, n]
    return th.mean(th.sum(D.min(1)[0], dim=1) / a.shape[1] + th.sum(D.min(2)[0], dim=1) / b.shape[1])


def _is_binary_1bit(image):
    return np.prod(image.shape) == np.sum(np.isin(image, [0, 1]))


def ensure_tensor(value):
    """Converts value to ndarray."""
    if value.ndim == 2:
        value = np.expand_dims(value, 0)
    return value


def _prepare_raster(raster, binarization=None, envelope=0):
    # ensure we are working with tensors of shape [n, h, w]
    raster = ensure_tensor(raster)

    # binarize the input raster images
    if binarization == 'maxink':
        binarization_func = color_utils.img_8bit_to_binary_maxink
    elif binarization == 'maxwhite':
        binarization_func = color_utils.img_8bit_to_binary_maxwhite
    elif binarization == 'median':
        binarization_func = color_utils.img_8bit_to_binary_median
    elif None is binarization:
        binarization_func = lambda image: image
    else:
        raise NotImplementedError

    raster = binarization_func(raster)
    assert _is_binary_1bit(raster), 'images are not binary (only values 0 and 1 allowed in images)'

    # invert images so that we compute a correct metric
    raster = 1 - raster

    # compute envelopes around ground-truth
    # to compute metrics with spatial tolerance
    if envelope:
        assert isinstance(envelope, int)
        raster = np.array([binary_dilation(r, iterations=envelope)
                           for r in raster]).astype(np.uint8)
    return raster


def raster_to_point_cloud(raster, binarization=None, envelope_true=0):
    '''
    Batch version of raster to point cloud
    :param raster: batch of raster images [...,h, w](example[b,1,h,w]) either from (0,255),or (0,1)
    :param binarization: raster image binarization type (maxink,maxwhite, median)
    :param envelope_true:
    :return: point cloud, [point_count,2+ 1 +  Indicator(batch)], where if batch parameter is presented every p[oint
    would be assigned to image number
    '''
    if (raster.max() > 1.):
        raster = _prepare_raster(raster, binarization=binarization, envelope=envelope_true)
    xyz = np.argwhere(raster > 0.5).astype(np.float64)
    if xyz.shape[1] == 2:
        xyz = np.concatenate((xyz, np.zeros((xyz.shape[0], 1))), axis=-1)
        xyz[:, 0] = xyz[:, 0] / raster.shape[0]
        xyz[:, 1] = xyz[:, 1] / raster.shape[1]
    else:
        xyz[:, -1] = xyz[:, -1] / raster.shape[-1]
        xyz[:, -2] = xyz[:, -2] / raster.shape[-2]
    return xyz


# def compute_distance_fields_for_image(images):
#     """
#
#     :param images: batch of grayscaler images [batch,1,height,width] numpy array
#     :return:
#     """
#     target_distance_fields = np.empty((images.shape[0], int(images.shape[-2]*1.5), int(images.shape[-1]*1.5)))
#     for t in range(images.shape[0]):
#         points = np.flip(raster_to_point_cloud(images[t], 'median')[:,1:],1)
#         grid = np.mgrid[-0.25:1.25:images.shape[-2]*1.5j, -0.25:1.25:images.shape[-1]*1.5j].T[:, :, None, :]
#         for i in range(grid.shape[0]):
#             for j in range(grid.shape[1]):
#                 target_distance_fields[t, i, j] = np.amin(np.linalg.norm(grid[i, j] - points, axis=1))
#
#     crop_i = int((-images.shape[-2]+grid.shape[0]-2)/2) ## worked only if image.shape %2==0
#     crop_j = int((-images.shape[-1] + grid.shape[1] - 2) / 2)
#     target_distance_fields = target_distance_fields[:,crop_i:-crop_i,crop_j:-crop_j].astype(np.float32)**2
#     target_alignment_fields = compute_alignment_fields(th.tensor(target_distance_fields))
#     target_distance_fields = target_distance_fields[:,1:-1, 1:-1]
#     target_occupancy_fields = compute_occupancy_fields(th.tensor(target_distance_fields))
#
#     return th.tensor(target_distance_fields), target_alignment_fields, target_occupancy_fields


def compute_distance_fields_for_image(images):
    """

    :param images: batch of grayscaler images [batch,1,height,width] numpy array
    :return:
    """
    target_distance_fields = np.empty((images.shape[0], int(images.shape[-2] * 1.5), int(images.shape[-1] * 1.5)))
    target_distance_fields_inv = np.empty((images.shape[0], int(images.shape[-2] * 1.5), int(images.shape[-1] * 1.5)))
    # print(images.max())
    # TODO ошибка max axis=1
    mask = (images > images.max() / 1.5).astype(int)
    mask = np.pad(mask, ((0, 0),
                         (0, 0), (int(0.25 * images.shape[2]), int(0.25 * images.shape[2])),
                         (int(0.25 * images.shape[3]), int(0.25 * images.shape[3]))), constant_values=1)
    # print(mask.mean())
    mask = np.squeeze(mask, axis=1)
    for t in range(images.shape[0]):
        points = np.flip(raster_to_point_cloud(images[t], 'median')[:, 1:], 1)
        points_inv = np.flip(raster_to_point_cloud(images[t].max() - images[t], 'median')[:, 1:], 1)
        grid = np.mgrid[-0.25:1.25:images.shape[-2] * 1.5j, -0.25:1.25:images.shape[-1] * 1.5j].T[:, :, None, :]
        grid_inv = np.mgrid[-0.25:1.25:images.shape[-2] * 1.5j, -0.25:1.25:images.shape[-1] * 1.5j].T[:, :, None, :]
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                target_distance_fields[t, i, j] = np.amin(np.linalg.norm(grid[i, j] - points, axis=1))
                target_distance_fields_inv[t, i, j] = np.amin(np.linalg.norm(grid_inv[i, j] - points_inv, axis=1))
    # deleted for 2 different alignment fields combination
    # target_distance_fields_inv = target_distance_fields * (mask) - target_distance_fields_inv * (1-mask)

    # target_distance_fields =target_distance_fields_inv
    crop_i = int((-images.shape[-2] + grid.shape[0] - 2) / 2)  ## worked only if image.shape %2==0
    crop_j = int((-images.shape[-1] + grid.shape[1] - 2) / 2)
    target_distance_fields = target_distance_fields[:, crop_i:-crop_i, crop_j:-crop_j].astype(np.float32) ** 2
    target_alignment_fields_inv = compute_alignment_fields(
        th.tensor(target_distance_fields_inv[:, crop_i:-crop_i, crop_j:-crop_j].astype(np.float32)))
    target_alignment_fields = compute_alignment_fields(
        th.tensor(target_distance_fields.astype(np.float32)))
    target_distance_fields = target_distance_fields[:, 1:-1, 1:-1]
    target_occupancy_fields = compute_occupancy_fields(th.tensor(target_distance_fields))
    # added for 2 different alignment fields combination

    mask = (target_occupancy_fields > 0).int()[..., None].repeat((1, 1, 1, 2))
    target_alignment_fields = target_alignment_fields * (1 - mask) + target_alignment_fields_inv * (mask)
    ##########################################################################################################
    return th.tensor(target_distance_fields), target_alignment_fields, target_occupancy_fields


# def compute_distance_fields_from_curves(curves, strokes, canvas_size=128):
#     '''
#     Compute distance field from curves
#     :param curves: curves parameters
#     :param strokes:
#     :param n_loops:
#     :param canvas_size:
#     :return:
#     '''
#     distance_fields = compute_distance_fields(curves, canvas_size)
#
#     distance_fields = th.max(distance_fields - strokes[..., None, None], th.zeros_like(distance_fields)).min(1)[0]
#     distance_fields = distance_fields ** 2
#
#     alignment_fields = compute_alignment_fields(distance_fields)
#     distance_fields = distance_fields[..., 1:-1, 1:-1]
#     occupancy_fields = compute_occupancy_fields(distance_fields)
#     return distance_fields, alignment_fields, occupancy_fields

def compute_distance_fields_from_curves(curves, strokes, n_loops=th.tensor([10]), topology=None,
                                        canvas_size=128):
    '''
    Compute distance field from curves
    :param curves: curves parameters
    :param strokes:
    :param n_loops:
    :param topology:
    :param canvas_size:
    :return:
    '''
    if topology is None:
        # TODO Ошибка ,нужно все таки переписать без loops
        topology = [2, 2, 2, 2, 2]
    distance_fields = compute_distance_fields(curves, canvas_size)
    # distance_fields = compute_distance_fields(curves, n_loops, topology, canvas_size)

    # distance_fields = th.max(distance_fields - strokes[..., None, None], th.zeros_like(distance_fields)).min(1)[0]
    int_loss = compute_intersection_loss(compute_alignment_fields(
        -(th.min(distance_fields - strokes[..., None, None], th.zeros_like(distance_fields)))),
                                         distance_fields - strokes[..., None, None])
    # print(int_loss)
    alignment_fields_inv = compute_alignment_fields(
        -(th.min(distance_fields - strokes[..., None, None], th.zeros_like(distance_fields))).min(1)[0])
    alignment_fields = compute_alignment_fields(
        th.max(distance_fields - strokes[..., None, None], th.zeros_like(distance_fields)).min(1)[0])

    distance_fields = th.max(distance_fields - strokes[..., None, None], th.zeros_like(distance_fields)).min(1)[0]
    # alignment_fields = compute_alignment_fields(distance_fields)
    distance_fields = distance_fields ** 2
    distance_fields = distance_fields[..., 1:-1, 1:-1]
    occupancy_fields = compute_occupancy_fields(distance_fields)

    mask = (occupancy_fields > 0).int()[..., None].repeat((1, 1, 1, 2))
    alignment_fields = alignment_fields * (1 - mask) + alignment_fields_inv * (mask)

    return distance_fields, alignment_fields, occupancy_fields, int_loss


def compute_intersection_loss(alignment_fields, distance_fields):
    ind_field = (distance_fields[..., 1:-1, 1:-1] < 0).long()
    # print(ind_field.shape)
    intersection_loss = 0
    for i in range(alignment_fields.shape[1]):
        for j in range(i + 1, alignment_fields.shape[1]):
            # print(th.sum(alignment_fields[:,i,...]*alignment_fields[:,j,...],dim=-1).shape)
            # I think here **2 was not needed,dunno
            intersection_loss += th.mean(
                th.sum(alignment_fields[:, i, ...] * alignment_fields[:, j, ...], dim=-1) ** 2 * \
                th.max(ind_field[:, i] + ind_field[:, j] - 1, th.zeros_like(ind_field[:, j])))
    return intersection_loss
