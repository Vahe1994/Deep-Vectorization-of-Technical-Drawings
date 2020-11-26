from numbers import Number
from typing import Callable, Iterable, Tuple

import numpy as np
import torch

import util_files.data.graphics_primitives as graphics_primitives
from util_files.rendering.cairo import render as _render, render_with_skeleton as _render_with_skeleton
# raise DeprecationWarning('vectran.train.diff_rendering is deprecated, '
#                          'use vectran.renderers.differentiable_rendering.sigmoids_renderer.renderer.Renderer')
# from vectran.train import diff_rendering


def _get_items_by_ids(loader, ids: Iterable):
    ids_sorting_ids = torch.argsort(torch.from_numpy(np.asarray(ids)))
    sorted_ids = ids[ids_sorting_ids]

    first_id_in_batch = 0
    current_outputs_id = 0
    stop = False
    for batch_i, (batch_inputs, batch_targets) in enumerate(loader):
        try:
            inputs
        except NameError:
            inputs = torch.empty(len(ids), *batch_inputs.shape[1:], dtype=batch_inputs.dtype,
                                 layout=batch_inputs.layout,
                                 device=batch_inputs.device, pin_memory=batch_inputs.is_pinned())
            targets = torch.empty(len(ids), *batch_targets.shape[1:], dtype=batch_targets.dtype,
                                  layout=batch_targets.layout,
                                  device=batch_targets.device, pin_memory=batch_targets.is_pinned())

        first_id_in_next_batch = first_id_in_batch + len(batch_inputs)
        while sorted_ids[current_outputs_id] < first_id_in_next_batch:
            id_in_batch = sorted_ids[current_outputs_id] - first_id_in_batch
            inputs[current_outputs_id] = batch_inputs[id_in_batch]
            targets[current_outputs_id] = batch_targets[id_in_batch]
            current_outputs_id += 1
            if current_outputs_id >= len(sorted_ids):
                stop = True
                break
        if stop: break
        first_id_in_batch = first_id_in_next_batch

    ids_back_to_original_order = torch.argsort(ids_sorting_ids)
    return inputs[ids_back_to_original_order].contiguous(), targets[ids_back_to_original_order].contiguous()


def get_ranked_subset_ids(scores: Iterable[Number], subset_size: int) -> (Iterable[int], Iterable[int], Iterable[int]):
    """Get ids of `subset_size` worst (lowest), best and closest to average `scores`."""
    scores = np.asarray(scores)
    worst_ids = np.argpartition(scores, subset_size)[:subset_size]
    best_ids = np.argpartition(-scores, subset_size)[:subset_size]
    average_ids = np.argpartition(np.abs(scores - scores.mean()), subset_size)[:subset_size]

    return worst_ids, best_ids, average_ids


def make_ranked_images_from_loader_and_model(model: Callable, loader, scores: Iterable[Number],
                                             imgrid_shape: Tuple[int, int], epoch_i=1, **kwargs) -> (
np.ndarray, np.ndarray, np.ndarray):
    images_n = np.prod(imgrid_shape)
    worst_ids, best_ids, average_ids = get_ranked_subset_ids(scores, images_n)
    all_ids = np.concatenate((worst_ids, best_ids, average_ids))
    inputs, targets = _get_items_by_ids(loader, all_ids)
    with torch.no_grad():
        outputs = model(inputs)
#     if epoch_i % 2 == 0:
#         y_pred = outputs.detach()
#         # y_pred_nowc = y_pred[..., :-2]
#         y_pred_nowc = y_pred[..., :]
#         y_pred_nowc.requires_grad = True


#         # optimizer = torch.optim.Adam([y_pred_nowc], betas=(0.9, 0.98), eps=1e-09, lr=0.001)
#         optimizer = torch.optim.SGD([y_pred_nowc], lr=0.1)
#         mse_loss = torch.nn.MSELoss()

#         for i in range(1500):
#             # Train for one batch

#             # y_pred_render = torch.cat((y_pred_nowc * 64, y_pred[..., -2:-1] * 64, y_pred[..., 5:]), dim=-1)
#             y_pred_render = torch.cat((y_pred_nowc[..., :5] * 64, y_pred_nowc[..., 5:]), dim=-1)

#             images_pred = diff_rendering.render(y_pred_render, (64, 64), sigmoid_rate=10)

#             loss = mse_loss(images_pred, inputs[:, 0, :, :])  # [B, C, H, W]

#             optimizer.zero_grad()

#             loss.backward()
#             optimizer.step()
#         #         y_pred[...,-2:] = width
#         outputs = y_pred
    outputs = outputs.detach().cpu().numpy()

    targets = targets.detach().cpu().numpy()

    lines_shape = (*imgrid_shape, *outputs.shape[-2:])

    worst_ids = slice(0, images_n)
    best_ids = slice(images_n, images_n * 2)
    average_ids = slice(images_n * 2, images_n * 3)

    worst_grid = make_images(outputs[worst_ids].reshape(lines_shape), targets[worst_ids].reshape(lines_shape), **kwargs)
    best_grid = make_images(outputs[best_ids].reshape(lines_shape), targets[best_ids].reshape(lines_shape), **kwargs)
    average_grid = make_images(outputs[average_ids].reshape(lines_shape), targets[average_ids].reshape(lines_shape),
                               **kwargs)

    return worst_grid, best_grid, average_grid


def make_ranked_images(lines_out, lines_gt, imgrid_shape, ranks, **kwargs):
    images_n = np.prod(imgrid_shape)

    worst_ids = np.argpartition(ranks, images_n)[:images_n]
    best_ids = np.argpartition(-ranks, images_n)[:images_n]
    average_ids = np.argpartition(np.abs(np.asarray(ranks) - np.mean(ranks)), images_n)[:images_n]

    lines_shape = (*imgrid_shape, *lines_out.shape[-2:])

    worst_grid = make_images(lines_out[worst_ids].reshape(lines_shape), lines_gt[worst_ids].reshape(lines_shape),
                             **kwargs)
    best_grid = make_images(lines_out[best_ids].reshape(lines_shape), lines_gt[best_ids].reshape(lines_shape), **kwargs)
    average_grid = make_images(lines_out[average_ids].reshape(lines_shape), lines_gt[average_ids].reshape(lines_shape),
                               **kwargs)

    return worst_grid, best_grid, average_grid


def make_images(lines_out, lines_gt, patch_size, patch_padding=(2, 2), patch_padding_color=230,
                stack_grids_horizontally=True, grid_padding=5, grid_padding_color=200, with_skeleton=True,
                skeleton_line_width=2, skeleton_node_size=8):
    # calculate dimensions and allocate image grid
    patch_width, patch_height = patch_size
    patches_n_vert, patches_n_hor = lines_out.shape[:2]
    padding_hor, padding_vert = patch_padding

    imgrid_width = (patch_width + padding_hor) * patches_n_hor - padding_hor
    imgrid_height = (patch_height + padding_vert) * patches_n_vert - padding_vert

    if stack_grids_horizontally:
        grid_origins = np.array([0, imgrid_width + grid_padding]) * np.arange(3)[..., None]
        imgrid_width = imgrid_width * 3 + grid_padding * 2
    else:
        grid_origins = np.array([imgrid_height + grid_padding, 0]) * np.arange(3)[..., None]
        imgrid_height = imgrid_height * 3 + grid_padding * 2

    channels_n = 3 if with_skeleton else 1

    imgrid = np.empty((imgrid_height, imgrid_width, channels_n), dtype=np.uint8)

    # fill paddings between patches
    for grid_origin_i, grid_origin_j in grid_origins:
        subgrid = imgrid[grid_origin_i:, grid_origin_j:]
        for patch_i in range(1, patches_n_vert):
            patch_origin_i = (patch_height + padding_vert) * patch_i
            subgrid[patch_origin_i - padding_vert: patch_origin_i] = patch_padding_color
        for patch_j in range(1, patches_n_hor):
            patch_origin_j = (patch_width + padding_hor) * patch_j
            subgrid[:, patch_origin_j - padding_hor: patch_origin_j] = patch_padding_color

        # fill paddings between grids
        for grid_origin_i, grid_origin_j in grid_origins[1:]:
            if stack_grids_horizontally:
                imgrid[:, grid_origin_j - grid_padding: grid_origin_j] = grid_padding_color
            else:
                imgrid[grid_origin_i - grid_padding: grid_origin_i] = grid_padding_color
    pt = graphics_primitives.PrimitiveType.PT_QBEZIER
    # substitute renderer
    if with_skeleton:
        renderer = lambda lines: render_with_skeleton(lines, patch_size=patch_size,
                                                      skeleton_line_width=skeleton_line_width,
                                                      skeleton_node_size=skeleton_node_size,
                                                      primitive_type=pt)
    else:
        renderer = lambda lines: render_without_skeleton(lines, patch_size=patch_size, primitive_type=pt)[..., None]

    # render outs
    grid_origin_i, grid_origin_j = grid_origins[0]
    render_lines_to(imgrid[grid_origin_i:, grid_origin_j:], lines_out, renderer, patch_size, patch_padding)

    # render gts
    grid_origin_i, grid_origin_j = grid_origins[1]
    render_lines_to(imgrid[grid_origin_i:, grid_origin_j:], lines_gt, renderer, patch_size, patch_padding)

    # render overlay
    if with_skeleton:
        renderer = lambda lines_out, lines_gt: render_overlay_colored(lines_out, lines_gt, patch_size, primitive_type=pt)
    else:
        renderer = lambda lines_out, lines_gt: render_overlay(lines_out, lines_gt, patch_size, primitive_type=pt)[..., None]
    grid_origin_i, grid_origin_j = grid_origins[2]
    render_line_pairs_to(imgrid[grid_origin_i:, grid_origin_j:], lines_out, lines_gt, renderer, patch_size,
                         patch_padding)

    if with_skeleton:
        return imgrid
    else:
        return imgrid.repeat(3, axis=-1)


def postprocess_primitives(primitives, linear_patch_size):
    drawn_primitives = primitives[primitives[..., -1] > .5]
    return drawn_primitives[:, :-1] * linear_patch_size


def render_with_skeleton(lines, patch_size, skeleton_line_width=2, skeleton_node_size=8,
                         primitive_type=graphics_primitives.PrimitiveType.PT_LINE):
    scaled_primitives = postprocess_primitives(lines, patch_size[0])
    # TODO fix
    return _render_with_skeleton({primitive_type: scaled_primitives}, patch_size,
                                 data_representation='vahe', line_width=skeleton_line_width,
                                 node_size=skeleton_node_size)


def render_without_skeleton(lines, patch_size, primitive_type=graphics_primitives.PrimitiveType.PT_LINE):
    scaled_primitives = postprocess_primitives(lines, patch_size[0])
    # TODO fix
    return _render({primitive_type: scaled_primitives}, patch_size,
                   data_representation='vahe')


def render_overlay(lines_out, lines_gt, patch_size, primitive_type=graphics_primitives.PrimitiveType.PT_LINE):
    scaled_out = postprocess_primitives(lines_out, patch_size[0])
    scaled_gt = postprocess_primitives(lines_gt, patch_size[0])
    # TODO fix
    out_image = _render({primitive_type: scaled_out}, patch_size, data_representation='vahe')
    gt_image = _render({primitive_type: scaled_gt}, patch_size, data_representation='vahe')

    return np.uint8(255) - ((np.uint8(255) - out_image) // np.uint8(2) + (np.uint8(255) - gt_image) // np.uint8(2))


def render_overlay_colored(lines_out, lines_gt, patch_size, primitive_type=graphics_primitives.PrimitiveType.PT_LINE):
    scaled_out = postprocess_primitives(lines_out, patch_size[0])
    scaled_gt = postprocess_primitives(lines_gt, patch_size[0])
    # TODO fix
    out_image = _render({primitive_type: scaled_out}, patch_size, data_representation='vahe')
    gt_image = _render({primitive_type: scaled_gt}, patch_size, data_representation='vahe')

    return np.dstack([out_image, gt_image, np.full_like(out_image, 255)])


def render_lines_to(imgrid, lines, renderer, patch_size, patch_padding):
    patch_width, patch_height = patch_size
    patches_n_vert, patches_n_hor = lines.shape[:2]
    padding_hor, padding_vert = patch_padding

    for patch_i in range(patches_n_vert):
        i_start = (patch_height + padding_vert) * patch_i
        for patch_j in range(patches_n_hor):
            j_start = (patch_width + padding_hor) * patch_j

            rendering = renderer(lines[patch_i, patch_j])
            rendering_h, rendering_w = rendering.shape[:2]
            imgrid[i_start: i_start + rendering_h, j_start: j_start + rendering_w] = rendering


def render_line_pairs_to(imgrid, lines_out, lines_gt, renderer, patch_size, patch_padding):
    patch_width, patch_height = patch_size
    patches_n_vert, patches_n_hor = lines_out.shape[:2]
    padding_hor, padding_vert = patch_padding

    for patch_i in range(patches_n_vert):
        i_start = (patch_height + padding_vert) * patch_i
        for patch_j in range(patches_n_hor):
            j_start = (patch_width + padding_hor) * patch_j

            rendering = renderer(lines_out[patch_i, patch_j], lines_gt[patch_i, patch_j])
            rendering_h, rendering_w = rendering.shape[:2]
            imgrid[i_start: i_start + rendering_h, j_start: j_start + rendering_w] = rendering