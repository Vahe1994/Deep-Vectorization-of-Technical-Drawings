from argparse import ArgumentParser
import os
import pickle
import sys

import numpy as np

sys.path.append('/code')
from util_files.job_tuples.calculate_results_for_curves import job_tuples
from util_files.simplification.join_qb import join_quad_beziers
from util_files.optimization.optimizer.logging import Logger
from util_files.rendering.cairo import PT_LINE, PT_QBEZIER
from util_files.data.graphics.graphics import Path, VectorImage


def main(options,vector_image_from_optimization=None, width_percentile=90, fit_tol=.5, w_tol=np.inf, join_tol=.5):
    logger = Logger.prepare_logger(loglevel='info', logfile=None)
    if vector_image_from_optimization is None:
        _, _, options.image_name, options.output_dir = job_tuples[options.dataset][options.job_id]
        sample_name = options.image_name[:-4]
        intermediate_output_path = f'{options.output_dir}/intermediate_output/{sample_name}.pickle'
        logger.info(f'1. Load intermediate output from {intermediate_output_path}')
        with open(intermediate_output_path, 'rb') as handle:
            intermediate_output = pickle.load(handle)

        for k, v in options.__dict__.items():
            setattr(intermediate_output['options'], k, v)
        options = intermediate_output['options']

        init_dir_name = 'random_initialization' if options.init_random else 'model_initialization'
        options.output_dir = f'{options.output_dir}/{init_dir_name}'

        optimization_output_path = f'{options.output_dir}/after_optimization/{sample_name}.svg'
        logger.info(f'2. Load vector image after optimization from {optimization_output_path}')
        vector_image = VectorImage.from_svg(optimization_output_path)
    else:
        sample_name = options.sample_name[:-4]
        vector_image = vector_image_from_optimization
    logger.info('3. Simplify curves')
    lines, curves = vector_image.vahe_representation()
    logger.info(f'\tinitial number of lines is {len(lines)}')
    logger.info(f'\tinitial number of curves is {len(curves)}')
    lines = np.asarray(lines)
    if len(curves) > 0:
        widths = np.asarray(curves)[..., -1]
        width_to_scale = np.percentile(widths, width_percentile)
        fit_tol = fit_tol * width_to_scale
        w_tol = w_tol * width_to_scale
        join_tol = join_tol * width_to_scale
        logger.info(f'\twidth to scale is {width_to_scale}')
        logger.info(f'\tfit_tol is {fit_tol}')
        logger.info(f'\tw_tol is {w_tol}')
        logger.info(f'\tjoin_tol is {join_tol}')
        curves = join_quad_beziers(curves, fit_tol=join_tol, w_tol=w_tol, join_tol=join_tol).numpy()
        logger.info(f'\tnumber of curves after merging is {len(curves)}')
    curves = np.asarray(curves)
    paths = ([Path.from_primitive(PT_QBEZIER, prim) for prim in curves] +
             [Path.from_primitive(PT_LINE, prim) for prim in lines])
    vector_image.paths = paths

    merging_output_path = f'{options.output_dir}/merging_output/{sample_name}.svg'
    logger.info(f'4. Save merging output to {merging_output_path}')
    os.makedirs(os.path.dirname(merging_output_path), exist_ok=True)
    vector_image.save(merging_output_path)
    return vector_image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['Precision Floor Plan', 'Golden', 'ABC',
                                                                       'ABC_gradual', 'cartoon'])
    parser.add_argument('--job_id', type=int, required=True)
    parser.add_argument('--init_random', action='store_true', help='init optimization randomly')

    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    main(options)