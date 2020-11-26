from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances

from util_files.data.graphics_primitives import GraphicsPrimitive, PT_LINE, repr_len_by_type, PT_QBEZIER
from util_files.rendering.cairo import render
import util_files.metrics.raster_metrics as r


VectorImage = Dict[str, List[GraphicsPrimitive]]


def _maybe_vector_to_raster(image, raster_res, data_representation='vahe'):
    if isinstance(image, Dict):
        raster = render(image, raster_res, data_representation)
    elif isinstance(image, List):
        raster = np.array([render(vector, raster_res, data_representation) for vector in image])
    elif isinstance(image, np.ndarray):
        raster = image
    else:
        raise TypeError('parameter image of unknown type')
    return raster


def hausdorff_score(image_true, image_pred, raster_res, **kwargs):
    """Computes Hausdorff metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.hausdorff_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.hausdorff_score(raster_true, raster_pred, **kwargs)


def psnr_score(image_true, image_pred, raster_res, **kwargs):
    """Computes PSNR metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.psnr_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.psnr_score(raster_true, raster_pred, **kwargs)


def f1_score(image_true, image_pred, raster_res, **kwargs):
    """Computes F1 metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.psnr_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.f1_score(raster_true, raster_pred, **kwargs)


def precision_score(image_true, image_pred, raster_res, **kwargs):
    """Computes Precision metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.precision_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.precision_score(raster_true, raster_pred, **kwargs)


def recall_score(image_true, image_pred, raster_res, **kwargs):
    """Computes Precision metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.recall_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.recall_score(raster_true, raster_pred, **kwargs)


def emd_score(image_true, image_pred, raster_res, **kwargs):
    """Computes IoU metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.iou_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.emd_score(raster_true, raster_pred, **kwargs)

def cd_score(image_true, image_pred, raster_res, **kwargs):
    """Computes IoU metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.iou_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.cd_score(raster_true, raster_pred, **kwargs)

def iou_score(image_true, image_pred, raster_res, **kwargs):
    """Computes IoU metric between ground-truth and predicted vectors.
    See `vectran.metrics.raster_metrics.iou_score` for reference."""
    raster_true = _maybe_vector_to_raster(image_true, raster_res)
    raster_pred = _maybe_vector_to_raster(image_pred, raster_res)
    return r.iou_score(raster_true, raster_pred, **kwargs)

# TODO sasha fix this PT_LINE should be default
def batch_numpy_to_vector(batch_numpy, raster_res, primitive_type=PT_QBEZIER):
    max_x, max_y = raster_res
    assert max_x == max_y
    l = repr_len_by_type[primitive_type] # TODO fix this
    really_lines = lambda item: round(item[l]) == 1.
    return [
        {primitive_type: np.array([item[:l] * max_x for item in filter(really_lines, array)])}
        for array in batch_numpy
    ]

# TODO fix PT_LINE should be default
def _vector_to_numpy(image, primitive_type=PT_QBEZIER):
    if isinstance(image, Dict):
        image_array = np.array(image[primitive_type])
    elif isinstance(image, List):
        image_array = np.array([vector[primitive_type] for vector in image])
    elif isinstance(image, np.ndarray):
        image_array = image
    else:
        raise TypeError('parameter image of unknown type')
    return image_array


def endpoint_score(image_true, image_pred, kind='full', metric='euclidean', average=None, **kwargs):
    """Computes metric between endpoints of ground-truth and predicted vectors."""

    def _metric(image_true, image_pred, kind='full', metric='euclidean'):
        if image_true.size == 0 or image_pred.size == 0:
            return np.nan

        image_true, image_pred = image_true[:4], image_pred[:4]
        cost_matrix = pairwise_distances(image_true, image_pred, metric=metric)

        if kind == 'full':  # compute max(N, M) correspondences
            pred_match_idx = cost_matrix.argmin(axis=1)  # indexes of preds that true primitives match to
            diffsq1 = cost_matrix[range(len(image_true)), pred_match_idx]

            true_match_idx = cost_matrix.argmin(axis=0)  # indexes of true primitives that preds match to
            left_pred_idx = np.array([idx for idx in range(len(image_pred)) if idx not in pred_match_idx]).astype(int)

            diffsq2 = cost_matrix[true_match_idx[left_pred_idx], left_pred_idx]
            diffsq = np.hstack((diffsq1, diffsq2))

        elif kind == 'bijection':  # compute min(N, M) correspondences (one-to-one matching)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            diffsq = cost_matrix[row_ind, col_ind]

        else:
            raise ValueError

        return np.mean(diffsq)


    tensor_true, tensor_pred = r.ensure_tensor(_vector_to_numpy(image_true)), r.ensure_tensor(_vector_to_numpy(image_pred))
    value = np.array([_metric(image_true, image_pred, kind=kind, metric=metric)
                     for image_true, image_pred in zip(tensor_true, tensor_pred)])
    value = value[~np.isnan(value)]
    if average == 'mean':
        value = np.mean(value, axis=0)

    return value


def mse_score(image_true, image_pred, kind='full', average=None, **kwargs):
    """Computes MSE metric between endpoints of ground-truth and predicted vectors."""
    return endpoint_score(image_true, image_pred, metric='euclidean', kind=kind, average=average, **kwargs)


def mae_score(image_true, image_pred, kind='full', average=None, **kwargs):
    """Computes MAE metric between endpoints of ground-truth and predicted vectors."""
    return endpoint_score(image_true, image_pred, metric='cityblock', kind=kind, average=average, **kwargs)


def nerror_score(image_true, image_pred, average=None, **kwargs):
    """Computes the stupid variant of the accuracy metric (how accurate is the number of predicted lines)
    between ground-truth and predicted vectors."""
    tensor_true, tensor_pred = _vector_to_numpy(image_true), _vector_to_numpy(image_pred)

    nerrors = [np.abs(len(image_true) - len(image_pred))
               for image_true, image_pred in zip(tensor_true, tensor_pred)]

    if average == 'mean':
        nerrors = np.mean(nerrors, axis=0)

    return nerrors


METRICS_BY_NAME = {
    'f1_score': f1_score,
    'precision_score': precision_score,
    'recall_score': recall_score,
    'iou_score': iou_score,
    # 'emd_score':emd_score,
    'cd_score': cd_score,
    'psnr_score': psnr_score,
    'hausdorff_score': hausdorff_score,
    'mse_score': mse_score,
    'mae_score': mae_score,
    'nerror_score': nerror_score,
}


__all__ = [
    'f1_score',
    'precision_score',
    'recall_score',
    'iou_score',
    'psnr_score',
    'cd_score',
    'emd_score',
    'hausdorff_score',
    'mse_score',
    'mae_score',
    'nerror_score',
    'batch_numpy_to_vector',
    'METRICS_BY_NAME',
]