import numpy as np
from scipy.ndimage.morphology import binary_dilation
from scipy.spatial.distance import directed_hausdorff
# import point_cloud_utils as pcu  # FIXME: remove pcu from the project


import util_files.color_utils as color_utils


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


def psnr_score(raster_true, raster_pred, max_db=60,
               average=None, envelope_true=0, binarization=None):
    """Computes PSNR metric between ground-truth and predicted images.

    :param raster_true: the input ground truth raster image
    :type raster_true: numpy.ndarray

    :param raster_pred: the input predicted raster image
    :type raster_pred: numpy.ndarray

    :param max_db: how much smoothing to apply when computing
        possibly zero quantities
    :type max_db: float

    :param average: If ``None``, does nothing. Otherwise, determines
    the kind of averaging to apply:
        ``'mean'``: computes mean values

    :param envelope_true: how much is the spatial tolerance
        to the inaccuracies in vectorization. The `raster_true`
        image will be dilated (i.e., its black lines expanded)
        by this many pixels in every direction.
    :type envelope_true: int

    :param binarization: If ``None``, no binarization is performed, and the images
        are assumed to be already binary. Otherwise, determines the type
        of binarization performed before computing metrics on 1-0 images:

        ``'maxink'``:
            Binarize the image, converting all grayscale values to black
            (thickening the lines, maximizing ink).

        ``'maxwhite'``:
            Binarize the image, converting all grayscale values to white
            (thinning the lines, maximizing whitespace).

        ``'median'``:
            Binarize the image, converting all values below 128 to black.

    :type binarization: str, optional

    :return: PSNR metric.
        Shape of the output tensor is determined by `average` variable.
        If `average` is None, all values are arrays of floats.
        If `average` is not None, all values are floating-point numbers.
    """
    raster_true = _prepare_raster(raster_true, binarization=binarization, envelope=envelope_true)
    raster_pred = _prepare_raster(raster_pred, binarization=binarization)

    smooth = 10 ** (-max_db // 10)
    axes = (1, 2)
    mse = ((raster_true - raster_pred) ** 2).mean(axis=axes)
    max_value = np.max(raster_true, axis=axes)
    psnr = 20 * np.log10(max_value) - 10 * np.log10(mse + smooth)
    if average == 'mean':
        psnr = np.mean(psnr, axis=0)

    return psnr


def precision_recall_fscore_iou_support(raster_true, raster_pred, beta=1.,
                                        smooth=1e-6, average=None, rtol=1e-8,
                                        envelope_true=0, binarization=None):
    """Computes precision, recall, F1 score, and Intersection over Union
    metrics all in one pass.

    :param raster_true: the input ground truth raster image
    :type raster_true: numpy.ndarray

    :param raster_pred: the input predicted raster image
    :type raster_pred: numpy.ndarray

    :param beta:
        The strength of recall versus precision in the F-score.
    :type beta: float

    :param smooth: how much smoothing to apply when computing
        possibly zero quantities
    :type smooth: float

    :param rtol: zeroes down values smaller than this
    :type rtol: float

    :param average: If ``None``, does nothing. Otherwise, determines
    the kind of averaging to apply:
        ``'mean'``: computes mean values

    :param envelope_true: how much is the spatial tolerance
        to the inaccuracies in vectorization. The `raster_true`
        image will be dilated (i.e., its black lines expanded)
        by this many pixels in every direction.
    :type envelope_true: int

    :param binarization: If ``None``, no binarization is performed, and the images
        are assumed to be already binary. Otherwise, determines the type
        of binarization performed before computing metrics on 1-0 images:

        ``'maxink'``:
            Binarize the image, converting all grayscale values to black
            (thickening the lines, maximizing ink).

        ``'maxwhite'``:
            Binarize the image, converting all grayscale values to white
            (thinning the lines, maximizing whitespace).

        ``'median'``:
            Binarize the image, converting all values below 128 to black.

    :type binarization: str, optional

    :return: precision, recall, F1 score, and IoU metrics.
        Shape of all measures is determined by `average` variable.
        If `average` is None, all values are arrays of floats.
        If `average` is not None, all values are floating-point numbers.
    """
    raster_true = _prepare_raster(raster_true, binarization=binarization, envelope=envelope_true)
    raster_pred = _prepare_raster(raster_pred, binarization=binarization)

    # compute true-positives, false-positives, and false-negatives
    axes = (1, 2)
    tp = np.sum(raster_true * raster_pred, axis=axes)
    fp = np.sum(raster_pred, axis=axes) - tp
    fn = np.sum(raster_true, axis=axes) - tp

    # compute target metrics: precision, recall, f_score, iou,
    # smoothing their values if required
    precision = (tp + smooth) / (tp + fp + smooth)

    recall = (tp + smooth) / (tp + fn + smooth)

    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall + smooth) / \
              (beta2 * precision + recall + smooth)
    f_score[np.isclose(recall, 0, rtol=rtol) & np.isclose(precision, 0, rtol=rtol)] = 0

    union = tp + fp + fn
    iou_score = (tp + smooth) / (union + smooth)

    # compute averaged versions if needed
    if average == 'mean':
        precision = np.mean(precision, axis=0)
        recall = np.mean(recall, axis=0)
        f_score = np.mean(f_score, axis=0)
        iou_score = np.mean(iou_score, axis=0)

    return precision, recall, f_score, iou_score


def f1_score(raster_true, raster_pred, beta=1,
             smooth=1e-6, average=None, rtol=1e-8,
             envelope_true=0, binarization=None):
    """Computes F1 score.
    See parameters for `precision_recall_fscore_iou_support`."""
    _, _, f1, _ = precision_recall_fscore_iou_support(raster_true, raster_pred,
                                                      beta=beta, smooth=smooth,
                                                      average=average, rtol=rtol,
                                                      envelope_true=envelope_true,
                                                      binarization=binarization)
    return f1


def iou_score(raster_true, raster_pred, beta=1,
              smooth=1e-6, average=None, rtol=1e-8,
              envelope_true=0, binarization=None):
    """Computes Intersection over Union score.
    See parameters for `precision_recall_fscore_iou_support`."""
    _, _, _, iou = precision_recall_fscore_iou_support(raster_true, raster_pred,
                                                       beta=beta, smooth=smooth,
                                                       average=average, rtol=rtol,
                                                       envelope_true=envelope_true,
                                                       binarization=binarization)
    return iou


# def raster_to_point_cloud(raster,binarization, envelope_true=0):
#     'Batch version.'
#     raster = _prepare_raster(raster, binarization=binarization, envelope=envelope_true)
#
#     xyz = np.argwhere(raster > 0.5).astype(np.float64)
#     if xyz.shape[1]==2:
#         xyz = np.concatenate((xyz,np.zeros((xyz.shape[0],1))),axis=-1)
#         xyz[:,0] = xyz[:,0]/raster.shape[0]
#         xyz[:,1] = xyz[:,1]/raster.shape[1]
#     else:
#         xyz[:,1] = xyz[:,1]/raster.shape[1]
#         xyz[:,2] = xyz[:,2]/raster.shape[2]
#     return xyz
#
#
# def emd_score(raster_true, raster_pred, beta=1,
#               smooth=1e-6, average=None, rtol=1e-8,
#               envelope_true=0, binarization=None):
#     """Computes earth moving distance score.Batch version.Input b,1,h,w. values 0,255"""
#     sinkhorn_dist = []
#     for it in range(raster_true.shape[0]):
#         xyz_true = raster_to_point_cloud(raster_true[it],binarization)
#         xyz_pred = raster_to_point_cloud(raster_pred[it], binarization)
#         if(xyz_true.shape[0]<=1 or xyz_pred.shape[0]<=1):
#             continue
#         M = pcu.pairwise_distances(xyz_true, xyz_pred)
#         w_a = np.ones(xyz_true.shape[0])
#         w_b = np.ones(xyz_pred.shape[0])
#
#         P = pcu.sinkhorn(w_a, w_b, M, eps=1e-3)
#
#         sinkhorn_dist.append((M * P).sum())
#     return np.mean(sinkhorn_dist)
#
#
# def cd_score(raster_true, raster_pred, beta=1,
#               smooth=1e-6, average=None, rtol=1e-8,
#               envelope_true=0, binarization=None):
#     """Computes chamfer distance score.Batch version."""
#     chamfer_dist = []
#     for it in range(raster_true.shape[0]):
#         xyz_true = raster_to_point_cloud(raster_true[it], binarization, envelope_true)
#         xyz_pred = raster_to_point_cloud(raster_pred[it], binarization)
# #         print(xyz_true.shape,xyz_pred.shape)
#         if(xyz_true.shape[0]<=1 or xyz_pred.shape[0]<=1):
#             continue
#         print(xyz_true.shape,xyz_pred.shape)
#         chamfer_dist.append(pcu.chamfer(xyz_true, xyz_pred))
#     return np.mean(chamfer_dist)




def precision_score(raster_true, raster_pred, beta=1,
                    smooth=1e-6, average=None, rtol=1e-8,
                    envelope_true=0, binarization=None):
    """Computes precision.
    See parameters for `precision_recall_fscore_iou_support`."""
    p, _, _, _ = precision_recall_fscore_iou_support(raster_true, raster_pred,
                                                       beta=beta, smooth=smooth,
                                                       average=average, rtol=rtol,
                                                       envelope_true=envelope_true,
                                                       binarization=binarization)
    return p


def recall_score(raster_true, raster_pred, beta=1,
                 smooth=1e-6, average=None, rtol=1e-8,
                 envelope_true=0, binarization=None):
    """Computes recall.
    See parameters for `precision_recall_fscore_iou_support`."""
    _, r, _, _ = precision_recall_fscore_iou_support(raster_true, raster_pred,
                                                     beta=beta, smooth=smooth,
                                                     average=average, rtol=rtol,
                                                     envelope_true=envelope_true,
                                                     binarization=binarization)
    return r


def hausdorff_score(raster_true, raster_pred, sample_size=1,
                    envelope_true=0, binarization=None, average=None):
    """Computes Hausdorff metric between ground-truth and predicted images.
    The measure is co_prepare_rastermputed by sampling the inky region of images.

    :param raster_true: the input ground truth raster image
    :type raster_true: numpy.ndarray

    :param raster_pred: the input predicted raster image
    :type raster_pred: numpy.ndarray

    :param sample_size: size of the sample, the denser the sample, the more
        precise the approximation
    :type sample_size: int

    :param average: If ``None``, does nothing. Otherwise, determines
    the kind of averaging to apply:
        ``'mean'``: computes mean values

    :param envelope_true: how much is the spatial tolerance
        to the inaccuracies in vectorization. The `raster_true`
        image will be dilated (i.e., its black lines expanded)
        by this many pixels in every direction.
    :type envelope_true: int

    :param binarization: If ``None``, no binarization is performed, and the images
        are assumed to be already binary. Otherwise, determines the type
        of binarization performed before computing metrics on 1-0 images:

        ``'maxink'``:
            Binarize the image, converting all grayscale values to black
            (thickening the lines, maximizing ink).

        ``'maxwhite'``:
            Binarize the image, converting all grayscale values to white
            (thinning the lines, maximizing whitespace).

        ``'median'``:
            Binarize the image, converting all values below 128 to black.

    :type binarization: str, optional

    :return: Hausdorff metric.
        Shape of the output tensor is determined by `average` variable.
        If `average` is None, all values are arrays of floats.
        If `average` is not None, all values are floating-point numbers.
    """
    raster_true = _prepare_raster(raster_true, binarization=binarization, envelope=envelope_true)
    raster_pred = _prepare_raster(raster_pred, binarization=binarization)

    # sample points from predicted and ground-truth primitives
    hausdorff_score = []
    for r_true, r_pred in zip(raster_true, raster_pred):
        # TODO update to filter out inf numbers when computing average
        coords_true = np.argwhere(r_true == 1)
        if 0 == len(coords_true):
            score = float('+inf')
        else:
            sample_idx_true = np.random.choice(len(coords_true), sample_size, replace=True)
            sample_true = coords_true[sample_idx_true]

        coords_pred = np.argwhere(r_pred == 1)
        if 0 == len(coords_pred):
            score = float('+inf')
        else:
            sample_idx_pred = np.random.choice(len(coords_pred), sample_size, replace=True)
            sample_pred = coords_pred[sample_idx_pred]

        if len(coords_true) > 0 and len(coords_pred) > 0:
            score = max(directed_hausdorff(sample_true, sample_pred)[0], directed_hausdorff(sample_pred, sample_true)[0])
        hausdorff_score.append(score)
    hausdorff_score = np.array(hausdorff_score)

    if average == 'mean':
        hausdorff_score = np.mean(hausdorff_score, axis=0)

    return hausdorff_score


__all__ = [
    'f1_score',
    'precision_score',
    'recall_score',
    'iou_score',
    # 'cd_score',
    # 'emd_score',
    'psnr_score',
    'hausdorff_score',
]