import numpy as np
from rtree import index
from merging.utils.merging_functions import tensor_vector_graph_numpy,merge_close,maximiz_final_iou,ordered,save_svg,lines_matching
import argparse
import torch

def postprocess(y_pred_render, patches_offsets, input_rgb, cleaned_image, it, options):
    '''

    :param y_pred_render: array after refinement of size [patch_number,number_of_lines,6]
    :param patches_offsets: array of size that [patches_number,2] containsinformaion about patches coordinate in image
    :param input_rgb: splitted image into patches
    :param cleaned_image: original clean image
    :param it: current image number
    :param options: dict with config
    :return:
    1)array of size [number_of line,6] with information of lines 2) rendered image after potprocessing
    '''
    nump = tensor_vector_graph_numpy(y_pred_render, patches_offsets, options)

    lines = nump.copy()
    lines = np.array(lines)
    lines = lines[lines[:, 1].argsort()[::-1]]
    idx = index.Index()
    widths = lines[:, 4]  # storing widths and coordinates separately
    ordered_lines = []
    for i, line in enumerate(lines):
        ordered_line = ordered(line)
        idx.insert(i, ordered_line)
        ordered_lines.append(ordered_line)

    result = np.array(merge_close(lines, idx, widths, max_angle=options.max_angle_to_connect, window_width=200,
                                  max_dist=options.max_angle_to_connect))
    save_svg(result, cleaned_image.shape, options.image_name[it], options.output_dir + 'merging_output/')
    result_tuning = np.array(maximiz_final_iou(result, input_rgb))
    save_svg(result_tuning, cleaned_image.shape, options.image_name[it], options.output_dir + 'iou_postprocess/')
    result_tuning = lines_matching(result_tuning, frac=0.07)
    tuned_image = save_svg(result_tuning, cleaned_image.shape, options.image_name[it], options.output_dir + 'lines_matching/')
    return result_tuning, tuned_image
