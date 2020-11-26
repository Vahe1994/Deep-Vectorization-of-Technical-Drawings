import numpy as np
import svgpathtools
import torch

from .chamferdist import ChamferDistance

from util_files.data.graphics.graphics import VectorImage


def sample_points(vector_image: VectorImage, points_n=None, point_density=None, dtype=np.float128):
    segments = [prim.segment for path in vector_image.paths for prim in path]
    lengths = np.array([seg.length() for seg in segments]).astype(dtype)
    cumsum = np.concatenate([[0], np.cumsum(lengths)])
    total_length = cumsum[-1]
    if points_n is None:
        points_n = int(np.ceil(point_density * total_length))
    pointcloud_arc_coordinates = np.linspace(0, total_length, points_n, dtype=dtype)
    fist_point_on_segment = np.concatenate(
        [np.argmax(np.array(pointcloud_arc_coordinates) >= cumsum[:-1, None], axis=1), [points_n]])
    points = np.concatenate([sample_points_at_arclengths(segments[seg_i], pointcloud_arc_coordinates[
                                                                          fist_point_on_segment[seg_i]:
                                                                          fist_point_on_segment[seg_i + 1]] - cumsum[
                                                             seg_i], dtype=dtype) for seg_i in range(len(segments))])
    return points


def sample_points_at_arclengths(segment, arclengths, dtype=np.float128, ilength_tol=1e-2):
    if isinstance(segment, svgpathtools.Line):
        start = segment.start
        start = np.array([start.real, start.imag])
        end = segment.end
        end = np.array([end.real, end.imag])
        return np.linspace(start, end, len(arclengths), dtype=dtype)
    elif isinstance(segment, (svgpathtools.QuadraticBezier, svgpathtools.CubicBezier)):
        ts = np.asarray(list(map(lambda s: segment.ilength(np.clip(s, 0, segment.length()), s_tol=ilength_tol), arclengths)), dtype=dtype)
        xs = np.poly1d(segment.poly().coeffs.real)(ts)
        ys = np.poly1d(segment.poly().coeffs.imag)(ts)
        return np.stack([xs, ys], axis=1)
    else:
        raise NotImplementedError(f'sample_points_at_arclengths is not implemented for {segment.__class__}')


def cpch_distance(vector_image1: VectorImage, vector_image2: VectorImage, points_per_unit_length_in_pixels=3,
                  sampling_dtype=np.float128):
    pc1 = sample_points(vector_image1, point_density=points_per_unit_length_in_pixels, dtype=sampling_dtype)
    pc2 = sample_points(vector_image2, point_density=points_per_unit_length_in_pixels, dtype=sampling_dtype)

    chamfer_distance = ChamferDistance()
    pc1 = torch.from_numpy(pc1[None].astype(np.float32)).cuda().contiguous()
    pc2 = torch.from_numpy(pc2[None].astype(np.float32)).cuda().contiguous()
    # add third dimension
    pc1 = torch.nn.functional.pad(pc1, [0, 1])
    pc2 = torch.nn.functional.pad(pc2, [0, 1])

    dist1_squared, dist2_squared, _, _ = chamfer_distance(pc1, pc2)
    del pc1, pc2

    return {'Chamfer distance in pixels squared': (dist1_squared.sum() + dist2_squared.sum()).item(),
            'Mean mean minimal distance in pixels': (dist1_squared.sqrt().mean() + dist2_squared.sqrt().mean()).item() / 2,
            'Hausdorff distance in pixels': torch.max(dist1_squared.max(), dist2_squared.max()).sqrt().item()}


def number_of_primitives(vector_image: VectorImage):
    return sum(len(path) for path in vector_image.paths)
