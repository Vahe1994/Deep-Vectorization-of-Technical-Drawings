from argparse import ArgumentParser
import os
import pickle
import sys
from time import time
import h5py
import numpy as np
import torch
from tqdm import trange

sys.path.append('/code')
from util_files.job_tuples.calculate_results_for_curves import job_tuples
from util_files.optimization.optimizer.logging import Logger
from util_files.evaluation_utils import vector_image_from_patches

from util_files.optimization.optimizer.adam import Adam
from util_files.optimization.primitives.line_tensor import LineTensor
from util_files.optimization.primitives.quadratic_bezier_tensor import QuadraticBezierTensor
from util_files.simplification.join_qb import join_quad_beziers



def main(options, intermediate_output=None, control_points_n=3, dtype=torch.float32, device='cuda',
         primitives_n=None, primitive_type=None, merge_period=60, lr=.05, the_width_percentile=90,
         optimization_iters_n   =None, batch_size=300, measure_period=20, reinit_period=20, max_reinit_iter=100,
         min_width=.3, min_confidence=64 * .5, min_length=1.7, append_iters_to_outdir=True):
    print(3)
    logger = Logger.prepare_logger(loglevel='info', logfile=None)
    print(3)
    if intermediate_output is not None:
        sample_name = options.sample_name[:-4]
        intermediate_output_path = f'{options.output_dir}/intermediate_output/{sample_name}.pickle'
        logger.info(f'1. Load intermediate output from {intermediate_output_path}')
    else:
        _, _, options.image_name, options.output_dir = job_tuples[options.dataset][options.job_id]
        sample_name = options.image_name[:-4]
        intermediate_output_path = f'{options.output_dir}/intermediate_output/{sample_name}.pickle'
        logger.info(f'1. Load intermediate output from {intermediate_output_path}')
        with open(intermediate_output_path, 'rb') as handle:
            intermediate_output = pickle.load(handle)
    for k, v in options.__dict__.items():
        setattr(intermediate_output['options'], k, v)
    options = intermediate_output['options']

    patch_offsets = torch.as_tensor(intermediate_output['patches_offsets'], dtype=dtype)
    model_output = torch.as_tensor(intermediate_output['patches_vector'], dtype=dtype)
    whole_image_size = intermediate_output['cleaned_image_shape']  # FIXME: cleaned_image_shape is different from actual svg size

    raster_patches = torch.as_tensor(intermediate_output['patches_rgb'], dtype=dtype)
    raster_patches = raster_patches.reshape(raster_patches.shape[:3])
    outp_dir = options.output_dir
    if not options.init_random:
        logger.info('2.5. Repatch')
        confident = model_output[:, :, -1] >= min_confidence
        widths = model_output[:, :, -2][confident].reshape(-1)
        the_width = np.percentile(widths, the_width_percentile)
        repatch_scale = int(round(the_width / 2))
        logger.info(f'\tthe width is {the_width}')
        logger.info(f'\tthe width percentile is {the_width_percentile}')
        logger.info(f'\trepatch scale is {repatch_scale}')
        raster_patches, patch_offsets, model_output = repatch(raster_patches, patch_offsets, model_output, repatch_scale)
        model_output = merge_close(model_output, min_confidence, the_width_percentile)
    else:
        repatch_scale = None

    patches_n, patch_height, patch_width = raster_patches.shape
    if primitives_n is None:
        primitives_n = model_output.shape[1]
    primitive_parameters_n = model_output.shape[2]

    logger.info(f'\t{patches_n} patches left with max {model_output.shape[1]} primitives per patch')

    # 2. Convert raster from uint8 0-255 with white background to 0-1 with 0 background
    raster_patches /= -255
    raster_patches += 1

    logger.info('3. Filter out empty patches')
    logger.info(f'\tfrom {patches_n} patches')
    nonempty = (raster_patches > 0).any(dim=-1).any(dim=-1)
    raster_patches = raster_patches[nonempty].contiguous()
    patch_offsets = patch_offsets[nonempty].contiguous()
    model_output = model_output[nonempty].contiguous()
    del nonempty
    patches_n = len(raster_patches)
    logger.info(f'\t{patches_n} patches left')

    if not options.init_random:
        logger.info('3.5. Sort patches')
        batches, patch_offsets = group_patches(model_output, raster_patches, patch_offsets, min_confidence,
                                               primitives_n, batch_size)
        patch_offsets = torch.cat(patch_offsets, dim=0)
        patches_n = len(patch_offsets)
        primitives_n = batches[-1][0].shape[1]
        logger.info(f'\t{patches_n} patches left with max {primitives_n} primitives per patch')
    else:
        batches = []
        for first_patch_i in range(0, patches_n, batch_size):
            next_first_patch_i = min(patches_n, first_patch_i + batch_size)
            batches.append(((next_first_patch_i - first_patch_i, primitives_n),
                            raster_patches[first_patch_i: next_first_patch_i]))
    del raster_patches

    # 4. Define the function to assemble patches into VectorImage
    def get_vectorimage(patches):
        return vector_image_from_patches(primitives=patches, patch_offsets=patch_offsets, image_size=whole_image_size,
                                         control_points_n=control_points_n, patch_size=[patch_width, patch_height],
                                         pixel_center_coodinates_are_integer=False, scale=repatch_scale,
                                         min_width=min_width, min_confidence=min_confidence, min_length=min_length)

    if (not options.init_random) and (not append_iters_to_outdir):
        model_output_path = f'{options.output_dir}/model_output/{sample_name}.svg'
        logger.info(f'5. Save model outputs to {model_output_path}')
        model_output = []
        for model_output_batch, _ in batches:
            pad = primitives_n - model_output_batch.shape[1]
            model_output.append(torch.nn.functional.pad(model_output_batch, [0, 0, 0, pad]))
        model_output = torch.cat(model_output, dim=0)
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        get_vectorimage(model_output).save(model_output_path)
    del model_output

    if optimization_iters_n is None:
        optimization_iters_n = options.optimization_iters_n

    init_dir_name = 'random_initialization' if options.init_random else 'model_initialization'
    if append_iters_to_outdir:
        options.output_dir = f'{options.output_dir}/{optimization_iters_n}/{init_dir_name}'
    else:
        options.output_dir = f'{options.output_dir}/{init_dir_name}'

    logger.info(f'6. Optimization parameters')
    if primitive_parameters_n == 6:
        primitive_type_from_model = 'lines'
    elif primitive_parameters_n == 8:
        primitive_type_from_model = 'qbeziers'
    else:
        raise NotImplementedError(f'Unknown number of parameters {primitive_parameters_n}')
    if primitive_type is None:
        primitive_type = primitive_type_from_model
    logger.info(f'\tprimitive type is {primitive_type}')
    logger.info(f'\tprimitive type of model outputs is {primitive_type_from_model}')

    if primitive_type == 'lines':
        primitive_parameters_n = 6
    elif primitive_type == 'qbeziers':
        primitive_parameters_n = 8
    if not options.init_random:
        primitives_after_optimization = batches[0][0].new_zeros([patches_n, primitives_n, primitive_parameters_n])
    else:
        primitives_after_optimization = torch.zeros([patches_n, primitives_n, primitive_parameters_n], dtype=dtype)

    init_primitives = make_init_primitives(options.init_random, primitive_type, batches, patch_width, patch_height,
                                           dtype, device, primitive_type_from_model, logger)

    metrics_file_path = f'{options.output_dir}/logs/{sample_name}.h5'
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
    logger.info(f'7. Prepare file with metrics at {metrics_file_path}')
    with h5py.File(metrics_file_path, 'w') as metrics_file:
        log_iters_n = (optimization_iters_n - 1) // measure_period + 1
        intersection_array = metrics_file.create_dataset('intersection', dtype='f', shape=[log_iters_n, patches_n])
        union_array = metrics_file.create_dataset('union', dtype='f', shape=[log_iters_n, patches_n])

        logger.info(f'8. Optimization')
        optimization_start_time = time()
        first_patch_i = 0
        for batch_i in trange(len(batches), file=logger.info_stream, desc='Optimize batches'):
            logger.info(f'\tInitialize')
            prim_ten = init_primitives(batch_i)
            q_raster = batches[batch_i][1]
            next_first_patch_i = first_patch_i + len(q_raster)
            aligner = Adam(prim_ten, q_raster, logger=logger, lr=lr)
            del prim_ten
            binary_raster = binarize(q_raster)
            del q_raster

            for i in trange(optimization_iters_n, desc=f'\toptimize patches {first_patch_i}-{next_first_patch_i - 1}',
                            file=logger.info_stream, position=0, leave=True):
                # 8.1. Measure and log metrics
                if i % measure_period == 0:
                    log_i = i // measure_period
                    rasterization = binarize(aligner.prim_ten.render_with_cairo_total(patch_width, patch_height,
                                                                                      min_width=min_width))
                    intersection_array[log_i, first_patch_i:next_first_patch_i] = (
                            rasterization & binary_raster).sum(dim=-1).sum(dim=-1)
                    union_array[log_i, first_patch_i:next_first_patch_i] = (
                            rasterization | binary_raster).sum(dim=-1).sum(dim=-1)
                    del rasterization

                if i % merge_period == 0:
                    aligner.prim_ten.merge_close()

                # 8.2. Optimize
                aligner.step(i, reinit_period=(reinit_period if i <= max_reinit_iter else 1000000))
            assemble_primitives_to(aligner.prim_ten, primitives_after_optimization,
                                   first_patch_i, next_first_patch_i, min_width, min_confidence)
            first_patch_i = next_first_patch_i
    logger.info(f'\tOptimization took {time() - optimization_start_time} seconds')
    del aligner, batches

    optimization_output_path = f'{options.output_dir}/after_optimization/{sample_name}.svg'
    logger.info(f'9. Save optimization result to {optimization_output_path}')
    os.makedirs(os.path.dirname(optimization_output_path), exist_ok=True)
    get_vectorimage(primitives_after_optimization).save(optimization_output_path)
    options.output_dir = outp_dir
    return primitives_after_optimization, patch_offsets, repatch_scale, get_vectorimage(primitives_after_optimization)


def repatch(raster_patches, patch_offsets, model_output, scale=None, h=None, w=None):
    if h is None:
        h = scale
    if w is None:
        w = scale

    patches_n = len(patch_offsets)
    patches_n_hor = (patch_offsets[:, 0] != 0).numpy().argmax()
    assert patches_n % patches_n_hor == 0
    patches_n_vert = len(patch_offsets) // patches_n_hor

    if patches_n_vert % h == 0:
        pad_vert = 0
    else:
        pad_vert = h - (patches_n_vert % h)
    if patches_n_hor % w == 0:
        pad_hor = 0
    else:
        pad_hor = w - (patches_n_hor % w)

    patch_h, patch_w = raster_patches.shape[1:]
    primitives_n, parameters_n = model_output.shape[1:]
    raster_patches = raster_patches.reshape(patches_n_vert, patches_n_hor, patch_h, patch_w)
    patch_offsets = patch_offsets.reshape(patches_n_vert, patches_n_hor, 2)
    model_output = model_output.reshape(patches_n_vert, patches_n_hor, primitives_n, parameters_n)

    raster_patches = torch.nn.functional.pad(raster_patches, [0, 0, 0, 0, 0, pad_hor, 0, pad_vert], value=255)
    patch_offsets = torch.nn.functional.pad(patch_offsets, [0, 0, 0, pad_hor, 0, pad_vert])
    model_output = torch.nn.functional.pad(model_output, [0, 0, 0, 0, 0, pad_hor, 0, pad_vert])

    patches_n_vert, patches_n_hor = raster_patches.shape[:2]
    patches_n_vert = patches_n_vert // h
    patches_n_hor = patches_n_hor // w

    raster_patches = raster_patches.reshape(patches_n_vert, h, patches_n_hor, w, patch_h, patch_w)
    patch_offsets = patch_offsets.reshape(patches_n_vert, h, patches_n_hor, w, 2)
    model_output = model_output.reshape(patches_n_vert, h, patches_n_hor, w, primitives_n, parameters_n)

    raster_patches = raster_patches.permute(0, 2, 1, 4, 3, 5)
    model_output = model_output.permute(0, 2, 1, 3, 4, 5)
    patch_offsets = patch_offsets[:, 0, :, 0]

    shifts = model_output.new_zeros([h, w, 1, parameters_n])
    for i in range(h):
        shifts[i, :, :, 1:-2:2] += i * patch_h
    for j in range(w):
        shifts[:, j, :, 0:-2:2] += j * patch_w

    model_output = model_output + shifts

    patch_h = patch_h * h
    patch_w = patch_w * w

    raster_patches = raster_patches.reshape(-1, patch_h, patch_w)
    model_output = model_output.reshape(-1, primitives_n * h * w, parameters_n)
    patch_offsets = patch_offsets.reshape(-1, 2)

    if scale is not None:
        assert (patch_h % scale == 0) and (patch_w % scale == 0)
        patch_h //= scale
        patch_w //= scale
        patch_offsets /= scale
        model_output[:, :, :-1] /= scale
        raster_patches = (raster_patches.reshape(-1, patch_h, scale, patch_w, scale)
                          .permute(0, 1, 3, 2, 4).mean(dim=[-1, -2]))

    raster_patches = raster_patches.contiguous()
    model_output = model_output.contiguous()
    patch_offsets = patch_offsets.contiguous()

    return raster_patches, patch_offsets, model_output


def group_patches(model_output, raster_patches, patches_offset, min_confidence, max_prims_n, max_batch_size):
    parameters_n = model_output.shape[2]
    confident_primitives = model_output[:, :, -1] >= min_confidence
    prims_n = confident_primitives.sum(dim=-1)

    batches = []
    patches_offset_batches = []
    for prims_n_in_batch in range(1, max_prims_n + 1):
        mask = prims_n == prims_n_in_batch
        if not mask.any():
            continue
        masked_model_out = model_output[mask]
        masked_model_out = masked_model_out[confident_primitives[mask], :].reshape(-1, prims_n_in_batch, parameters_n)
        masked_raster_patches = raster_patches[mask]
        masked_patches_offset = patches_offset[mask]
        patches_n = len(masked_model_out)
        for first_patch_i in range(0, patches_n, max_batch_size):
            next_first_patch_i = min(first_patch_i + max_batch_size, patches_n)
            batches.append([masked_model_out[first_patch_i:next_first_patch_i],
                            masked_raster_patches[first_patch_i:next_first_patch_i]])
            patches_offset_batches.append(masked_patches_offset[first_patch_i:next_first_patch_i])
    return batches, patches_offset_batches


def make_init_primitives(random, primitive_type, batches, patch_width, patch_height, dtype, device,
                         primitive_type_from_model, logger):
    if random:
        logger.info(f'\tinitialization is random')
        if primitive_type == 'lines':
            def init_primitives(batch_i):
                n, primitives_n = batches[batch_i][0]
                return LineTensor(
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 1, primitives_n) + 1, dtype=dtype, device=device)
        elif primitive_type == 'qbeziers':
            def init_primitives(batch_i):
                n, primitives_n = batches[batch_i][0]
                return QuadraticBezierTensor(
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 2, primitives_n) * torch.tensor([[patch_width], [patch_height]], dtype=dtype),
                    torch.rand(n, 1, primitives_n) + 1, dtype=dtype, device=device)
    else:
        logger.info(f'\tinitialization is model')
        if primitive_type == 'lines':
            if primitive_type_from_model == 'lines':
                def init_primitives(batch_i):
                    model_output = batches[batch_i][0].permute(0, 2, 1).contiguous()
                    p1 = model_output[:, :2]
                    p2 = model_output[:, 2:4]
                    w = model_output[:, 4:5]
                    return LineTensor(p1, p2, w, dtype=dtype, device=device)
            else:
                raise NotImplementedError('Please implement conversion from curves to lines')
        elif primitive_type == 'qbeziers':
            if primitive_type_from_model == 'qbeziers':
                def init_primitives(batch_i):
                    model_output = batches[batch_i][0].permute(0, 2, 1).contiguous()
                    p1 = model_output[:, :2]
                    p2 = model_output[:, 2:4]
                    p3 = model_output[:, 4:6]
                    w = model_output[:, 6:7]
                    return QuadraticBezierTensor(p1, p2, p3, w, dtype=dtype, device=device)
            elif primitive_type_from_model == 'lines':
                def init_primitives(batch_i):
                    model_output = batches[batch_i][0].permute(0, 2, 1).contiguous()
                    p1 = model_output[:, :2]
                    p3 = model_output[:, 2:4]
                    p2 = (p1 + p3) / 2
                    w = model_output[:, 4:5]
                    return QuadraticBezierTensor(p1, p2, p3, w, dtype=dtype, device=device)
    return init_primitives


def assemble_primitives_to(primitive_tensor, data_tensor, first_patch_i, next_first_patch_i, min_width, min_confidence):
    primitives_n = primitive_tensor.primitives_n
    good_confidence = min_confidence * 2
    if isinstance(primitive_tensor, LineTensor):
        p1 = primitive_tensor.p1.data.cpu()
        p2 = primitive_tensor.p2.data.cpu()
        width = primitive_tensor.width.data.cpu()
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, :2] = p1.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 2:4] = p2.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 4] = width[:, 0]
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 5] = width[:, 0] >= min_width
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 5] *= good_confidence
    elif isinstance(primitive_tensor, QuadraticBezierTensor):
        p1 = primitive_tensor.p1.data.cpu()
        p2 = primitive_tensor.p2.data.cpu()
        p3 = primitive_tensor.p3.data.cpu()
        width = primitive_tensor.width.cpu()
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, :2] = p1.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 2:4] = p2.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 4:6] = p3.permute(0, 2, 1)
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 6] = width[:, 0]
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 7] = width[:, 0] >= min_width
        data_tensor[first_patch_i:next_first_patch_i, :primitives_n, 7] *= good_confidence


def binarize(raster):
    return raster > .5


def merge_close(model_output, min_confidence, width_percentile=90):
    patches_n, primitives_n, parameters_n = model_output.shape
    new_model_output = model_output.new_zeros([patches_n, primitives_n, parameters_n])
    max_primitives_in_patch = 0
    for i, patch in enumerate(model_output):
        confident = patch[:, -1] >= min_confidence
        if not confident.any():
            continue
        patch = patch[confident, :-1]
        common_width_in_patch = np.percentile(patch[:, -1], width_percentile)
        join_tol = .5 * common_width_in_patch
        fit_tol = .5 * common_width_in_patch
        w_tol = np.inf
        new_patch = join_quad_beziers(patch, join_tol=join_tol, fit_tol=fit_tol, w_tol=w_tol)
        new_patch = np.pad(new_patch, [[0, 0], [0, 1]], constant_values=min_confidence * 2)
        new_primitives_n = len(new_patch)
        max_primitives_in_patch = max(max_primitives_in_patch, new_primitives_n)
        new_model_output.numpy()[i, :new_primitives_n] = new_patch
    return new_model_output[:, :max_primitives_in_patch]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['Precision Floor Plan', 'Golden', 'ABC', 'cartoon'])
    parser.add_argument('--job_id', type=int, required=True)
    parser.add_argument('--optimization_iters_n', type=int, default=100, help='iteration count')
    parser.add_argument('--init_random', action='store_true', help='init optimization randomly')

    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    main(options)