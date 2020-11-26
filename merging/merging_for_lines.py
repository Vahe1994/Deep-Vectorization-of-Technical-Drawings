import numpy as np
from rtree import index
from merging.utils.merging_for_lines_functions import tensor_vector_graph_numpy,merge_close,maximiz_final_iou,ordered,save_svg,lines_matching,index
import argparse
import torch

def postprocess(y_pred_render, patches_offsets, input_rgb, cleaned_image, it, options):
    '''

    :param y_pred_render:
    :param patches_offsets:
    :param input_rgb:
    :param cleaned_image:
    :param it:
    :param options:
    :return:
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
    #     result_tuning = lines

    result = np.array(merge_close(lines, idx, widths, max_angle=options.max_angle_to_connect, window_width=200,
                                  max_dist=options.max_angle_to_connect))
    save_svg(result, cleaned_image.shape, options.image_name[it], options.output_dir + 'merging_output/')
    #     result_tuning = result
    result_tuning = np.array(maximiz_final_iou(result, input_rgb))
    print(result_tuning.shape)
    save_svg(result_tuning, cleaned_image.shape, options.image_name[it], options.output_dir + 'iou_postprocess/')
    result_tuning = lines_matching(result_tuning, frac=0.07)
    save_svg(result_tuning, cleaned_image.shape, options.image_name[it], options.output_dir + 'lines_matching/')


    return result_tuning
