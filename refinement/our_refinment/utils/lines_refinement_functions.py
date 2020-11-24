from abc import ABC, abstractmethod
import os
import signal
import sys

from IPython.display import display, clear_output
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch

from util_files.rendering.cairo import render as original_render, render_with_skeleton as original_render_with_skeleton, \
    PT_LINE

import math
import torch
import sys

from tqdm import tqdm
### Todo check this metric or change it
from util_files.metrics.raster_metrics import iou_score

h, w = 64, 64
padding = 3
padded_h = h + padding*2
padded_w = w + padding*2
device = device = torch.device('cuda')

pixel_center_coodinates_are_integer = False
dtype =torch.float32
raster_coordinates = torch.meshgrid(torch.arange(-padding, h+padding, dtype=dtype), torch.arange(-padding, w+padding, dtype=dtype))
raster_coordinates = torch.stack([raster_coordinates[1].reshape(-1), raster_coordinates[0].reshape(-1)])

if not pixel_center_coodinates_are_integer:
    raster_coordinates += .5
raster_coordinates = raster_coordinates.to(device)





class NonanAdam(torch.optim.Adam):
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad[~torch.isfinite(grad)] = 0  ### <---
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                assert np.isfinite(step_size)
                assert torch.all(torch.isfinite(exp_avg))
                assert torch.all(torch.isfinite(denom))

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


def render(data, dimensions, data_representation):
    return original_render(data, dimensions, data_representation, linecaps='butt', linejoin='miter')


def render_skeleton(data, dimensions, data_representation, padding=0, scaling=4):
    w, h = dimensions
    data[..., [0, 2]] = np.clip(data[..., [0, 2]], .5, w - .5)
    data[..., [1, 3]] = np.clip(data[..., [1, 3]], .5, h - .5)
    data[..., :4] += padding
    data = data * [scaling, scaling, scaling, scaling, 1]
    data[..., :4][data[..., -1] < 1 / 4] = -100
    data[..., -1] = 0
    return original_render_with_skeleton({PT_LINE: data}, [(w + padding * 2) * scaling, (h + padding * 2) * scaling],
                                         data_representation, linecaps='butt', linejoin='miter', line_color=(0, 0, 0),
                                         line_width=2)


def get_random_line(h, w, max_width=None):
    if max_width is None:
        max_width = max(h, w) // 10
    line = np.float32(np.random.random(5) * [w, h, w, h, max_width])
    line[..., -1] += 1
    return line


class SupersamplingStrategy(ABC):
    @abstractmethod
    def supersample(self, raster_coordinates):
        """Generate coordinates of supersamples for given coordinates of rasters."""
        pass

    def subsample(self, supersamples):
        """Average supersamples to get the actual raster"""
        pass

    def reorder_supersamples(self, supersamples):
        """Reorder supersamples, generated for C ordered (row major) raster_coordinates, for being in C order.

        Parameters
        ----------
        supersamples : torch.Tensor
            Tensor of shape [**, h, w]
        """
        pass


class RegularSupersampling(SupersamplingStrategy):
    def __init__(self, supersamples_per_dimension=4):
        self.supersamples_per_dimension = supersamples_per_dimension

    def supersample(self, raster_coordinates):
        """Generate coordinates of supersamples for given coordinates of rasters."""
        n = self.supersamples_per_dimension
        rasters_n = raster_coordinates.shape[1]
        n2 = n ** 2
        supersample_coordinates = torch.empty([2, rasters_n * n2], dtype=raster_coordinates.dtype,
                                              device=raster_coordinates.device)
        for i in range(n):
            xs = raster_coordinates[0] + (i / n - (n - 1) / (n * 2))
            ys = raster_coordinates[1] + (i / n - (n - 1) / (n * 2))
            for j in range(n):
                supersample_coordinates[0, i + n * j:: n2] = xs
                supersample_coordinates[1, i * n + j:: n2] = ys
        del xs, ys
        return supersample_coordinates

    def subsample(self, supersamples, dtype=torch.float32):
        """Average supersamples to get the actual raster"""
        return torch.nn.functional.avg_pool1d(supersamples.type(dtype), self.supersamples_per_dimension ** 2)

    def reorder_supersamples(self, supersamples):
        """Reorder supersamples, generated for C ordered (row major) raster_coordinates, for being in C order."""
        shape = supersamples.shape
        other_dims = shape[:-2]
        H = W = self.supersamples_per_dimension
        hH, wW = shape[-2:]
        assert (hH % H == 0) and (wW % W == 0)
        h = hH // H
        w = wW // W
        supersamples = supersamples.reshape(*other_dims, h, w, H, W).transpose(-3, -2)
        return supersamples.reshape(shape)


class RGSSSupersampling(SupersamplingStrategy):
    """Only 4xRGSS by now."""

    def supersample(self, raster_coordinates):
        """Generate coordinates of supersamples for given coordinates of rasters."""
        rasters_n = raster_coordinates.shape[1]
        supersample_coordinates = torch.empty([2, rasters_n * 4], dtype=raster_coordinates.dtype,
                                              device=raster_coordinates.device)
        supersample_coordinates[0, 0::4] = raster_coordinates[0] + 1 / 8
        supersample_coordinates[1, 0::4] = raster_coordinates[1] - 3 / 8
        supersample_coordinates[0, 1::4] = raster_coordinates[0] - 1 / 8
        supersample_coordinates[1, 1::4] = raster_coordinates[1] - 1 / 8
        supersample_coordinates[0, 2::4] = raster_coordinates[0] + 3 / 8
        supersample_coordinates[1, 2::4] = raster_coordinates[1] + 1 / 8
        supersample_coordinates[0, 3::4] = raster_coordinates[0] - 1 / 8
        supersample_coordinates[1, 3::4] = raster_coordinates[1] + 3 / 8
        return supersample_coordinates

    def subsample(self, supersamples, dtype=torch.float32):
        """Average supersamples to get the actual raster"""
        return torch.nn.functional.avg_pool1d(supersamples.type(dtype), 4)

    def reorder_supersamples(self, supersamples):
        """Reorder supersamples, generated for C ordered (row major) raster_coordinates, for being in C order."""
        raise NotImplementedError


def render_lines(x1, y1, x2, y2, width, sample_coordinates, samples=None, linecaps='butt',
                 dtype=torch.float16, requires_grad=False, sparse=False,division_epsilon = 1e-12):
    """...
    Requires 34 bytes of GPU memory per patch per line per sample for half precision computations (default),
             45 for single precision computations, and 65 for double presision computations,
     i.e 2.7/3.5/5 GB to render 128 patches of 10 lines in patches of size 64x64 with 4x4 supersampling.

    Parameters
    ----------
        samples : torch.BoolTensor
    """
    # assert linecaps == 'butt', 'Others are not implemented by now (but easy to implement)'
    # assert not requires_grad, 'Not implemented'
    batch_size, lines_n = x1.shape
    rasters_n = sample_coordinates.shape[1]

    with torch.set_grad_enabled(requires_grad):
        l_dir = torch.empty(2, batch_size, lines_n, dtype=dtype, device=x1.device)
        l_dir[0] = x2.type(dtype)
        l_dir[0] -= x1.type(dtype)
        l_dir[1] = y2.type(dtype)
        l_dir[1] -= y1.type(dtype)
        length = torch.norm(l_dir, dim=0)
        l_dir /= (length + division_epsilon)

        R = torch.empty(2, batch_size, lines_n, rasters_n, dtype=l_dir.dtype, device=l_dir.device)
        R[0] = sample_coordinates[0]
        R[0] -= x1[..., None].type(R.dtype)
        R[1] = sample_coordinates[1]
        R[1] -= y1[..., None].type(R.dtype)

        proj = l_dir[0, ..., None] * R[0]
        proj += l_dir[1, ..., None] * R[1]

        shaded = proj <= length[..., None]
        shaded &= proj >= 0
        del length

        dist = proj
        dist[:] = l_dir[0, ..., None] * R[1]
        dist -= l_dir[1, ..., None] * R[0]
        dist.abs_()

        half_width = (width / 2).type(dtype)
        shaded &= dist <= half_width[..., None]
        del dist, proj, R, l_dir, half_width

        if sparse:
            ids = torch.nonzero(shaded).t().contiguous()
            del shaded
            if samples is None:
                samples = torch.sparse.ByteTensor(ids, torch.ones(1, dtype=torch.uint8, device=width.device).expand(
                    ids.shape[1]))
                del ids
                return samples
            else:
                pass  # TODO
        else:
            if samples is None:
                return shaded
            else:
                samples += shaded
                del shaded

supersampling_strategy = RegularSupersampling(4)
sample_coordinates = supersampling_strategy.supersample(raster_coordinates)

def render_lines_pt(lines_batch, samples=None, uint=False):
    rasters = render_lines(lines_batch[:, :, 0], lines_batch[:, :, 1], lines_batch[:, :, 2], lines_batch[:, :, 3],
                           lines_batch[:, :, 4], sample_coordinates, samples=samples)
    # sum all lines
    rasters = rasters.sum(1, keepdim=True, dtype=rasters.dtype)
    rasters = supersampling_strategy.subsample(rasters).reshape(-1, padded_h, padded_w)
    if uint:
        return np.uint8(torch.clamp((1 - rasters) * 255, 0, 255).detach().cpu().numpy())
    else:
        return rasters


def render_lines_skeleton(lines_batch):
    return np.stack(list(map(lambda lines: render_skeleton(lines, [w, h], data_representation='vahe', padding=padding),
                             lines_batch.detach().cpu().numpy())))


def line_to_point_energy_canonical(line_length, line_halfwidth, point_x, point_y, R):
    r"""Gives the total energy of interaction of a line and a point
     for a point-to-point interaction energy given by `-exp(-r**2/R**2)`.
    Coordinates of the point `point_x` and `point_y` correspond to the coordiante system
     where one end of the line is in the origin and the other is on the positive side of y axis.
    """
    return (
                   torch.erf((line_length - point_y) / R) + torch.erf(point_y / R)
           ) * (
                   torch.erf((line_halfwidth - point_x) / R) + torch.erf((line_halfwidth + point_x) / R)
           ) * (R ** 2)


def line_to_point_energy(lines_batch, point_charges,division_epsilon = 1e-12,R_close = 1,R_far = 32,far_weight = 1 / 50):
    r"""For each line in `lines_batch` gives the total energy of interaction of this line with `point_charges`.

    Parameters
    ----------
    lines_batch : torch.Tensor
        of shape [batch_size, lines_n, params_n]

    point_charges : torch.Tensor
        of shape [batch_size, lines_n, rasters_n] or [**, rasters_n] broadcastable to this shape
    """
    close_weight = 1 - far_weight

    batch_size, lines_n = lines_batch.shape[:2]
    rasters_n = point_charges.shape[-1]

    # Get parameters of the lines
    x1 = lines_batch[..., 0]
    y1 = lines_batch[..., 1]
    x2 = lines_batch[..., 2]
    y2 = lines_batch[..., 3]
    half_width = lines_batch[..., 4] / 2
    lx = x2 - x1
    ly = y2 - y1
    length = torch.sqrt(lx ** 2 + ly ** 2)
    nonzero_length = torch.max(length, torch.full([1], division_epsilon, dtype=length.dtype, device=length.device))
    lx = lx / nonzero_length
    ly = ly / nonzero_length
    del lines_batch, x2, y2, nonzero_length

    # Broadcast tensors
    x1, y1, half_width, lx, ly, length, point_charges, raster_x, raster_y = torch.broadcast_tensors(x1.unsqueeze(-1),
                                                                                                    y1.unsqueeze(-1),
                                                                                                    half_width.unsqueeze(
                                                                                                        -1),
                                                                                                    lx.unsqueeze(-1),
                                                                                                    ly.unsqueeze(-1),
                                                                                                    length.unsqueeze(
                                                                                                        -1),
                                                                                                    point_charges,
                                                                                                    raster_coordinates[
                                                                                                        0],
                                                                                                    raster_coordinates[
                                                                                                        1])

    # Translate points to canonical coordinate systems of the lines
    translated_raster_x = raster_x - x1
    translated_raster_y = raster_y - y1
    del raster_x, raster_y, x1, y1
    canonical_raster_x = translated_raster_x * ly - translated_raster_y * lx
    canonical_raster_y = translated_raster_x * lx + translated_raster_y * ly
    del translated_raster_x, translated_raster_y, ly, lx

    # Calculate energies per line
    energies = line_to_point_energy_canonical(length, half_width, canonical_raster_x, canonical_raster_y,
                                              R_close) * close_weight \
               + line_to_point_energy_canonical(length, half_width, canonical_raster_x, canonical_raster_y,
                                                R_far) * far_weight
    del canonical_raster_x, canonical_raster_y, length, half_width
    energies = torch.sum(energies * point_charges, dim=-1)
    del point_charges
    return energies


def line_to_vector_energy(lines_batch, vector_field,division_epsilon= 1e-12,R_close = 1):
    r"""For each line in `lines_batch` gives the total energy of interaction of this line with `vector_field`.

    Parameters
    ----------
    lines_batch : torch.Tensor
        of shape [batch_size, lines_n, params_n]

    vector_field : torch.Tensor
        of shape [2, batch_size, lines_n, rasters_n]
    """

    collinearity_beta = 1 / ((np.abs(np.cos(15 * np.pi / 180)) - 1) ** 2)
    collinearity_field_weight = 2

    batch_size, lines_n = lines_batch.shape[:2]
    rasters_n = vector_field.shape[-1]

    # Get parameters of the lines
    x1 = lines_batch[..., 0]
    y1 = lines_batch[..., 1]
    x2 = lines_batch[..., 2]
    y2 = lines_batch[..., 3]
    half_width = lines_batch[..., 4] / 2
    lx = x2 - x1
    ly = y2 - y1
    length = torch.sqrt(lx ** 2 + ly ** 2)
    nonzero_length = torch.max(length, torch.full([1], division_epsilon, dtype=length.dtype, device=length.device))
    lx = lx / nonzero_length
    ly = ly / nonzero_length
    del lines_batch, x2, y2, nonzero_length

    # Broadcast tensors
    x1, y1, half_width, lx, ly, length, raster_x, raster_y = torch.broadcast_tensors(x1.unsqueeze(-1), y1.unsqueeze(-1),
                                                                                     half_width.unsqueeze(-1),
                                                                                     lx.unsqueeze(-1), ly.unsqueeze(-1),
                                                                                     length.unsqueeze(-1),
                                                                                     raster_coordinates[0],
                                                                                     raster_coordinates[1])

    # Calculate intensities of vector field interactions and total field interaction intensities
    vector_field_norm = torch.norm(vector_field, dim=0)
    vector_field_norm += division_epsilon
    vector_field /= vector_field_norm
    vector_field_interaction_intensities = vector_field[0] * lx + vector_field[1] * ly
    del vector_field
    vector_field_interaction_intensities = torch.exp(
        -(vector_field_interaction_intensities.abs() - 1) ** 2 * collinearity_beta) * vector_field_norm
    del vector_field_norm

    # Translate points to canonical coordinate systems of the lines
    translated_raster_x = raster_x - x1
    translated_raster_y = raster_y - y1
    del raster_x, raster_y, x1, y1
    canonical_raster_x = translated_raster_x * ly - translated_raster_y * lx
    canonical_raster_y = translated_raster_x * lx + translated_raster_y * ly
    del translated_raster_x, translated_raster_y, ly, lx

    # Calculate energies per line
    energies = line_to_point_energy_canonical(length, half_width, canonical_raster_x, canonical_raster_y, R_close)
    del canonical_raster_x, canonical_raster_y, length, half_width
    energies = torch.sum(energies * vector_field_interaction_intensities, dim=-1)
    del vector_field_interaction_intensities
    return energies * collinearity_field_weight


def mean_field_energy_lines(lines_batch, rasters_batch, empty_charge=0, close_range_weight=2 * (1 / .5),
                            elementary_halfwidth=1 / 2, visibility_padding=2,division_epsilon= 1e-12):
    r"""...
    Algorithm is (for each batch):
    1. Render each line on binary supersample grid
    2. Sum (OR) individual renderings -- this is total positive charge field
    3. For each line calculate the sum from step 2 minus the rendering of this line from step 1
       -- this is the excess positive charge field for this line
    4. Subsample the renderings from step 3
    5. Subtract the actual raster from the subsampled renderings from step 4
       -- this is the excess charge field for each line

    Steps 6-8 are needed to avoid local minima.

    6. For each line calculate coordinates of each excess charge in the coordinate system of this line,
       where the y axis is aligned along the length, the x axis is aligned along the width,
       and the origin is in the center of the line
    7. For each line find the largest rectangle aligned along this line and filled with nonempty pixels only
    7..Such rectangle can be non unique, so define it like this:
    7.1. Select all empty pixels within `elementary_halfwidth` around the direction of the line, i.e |x| <= `elementary_halfwidth`
    7.2. Find the  pixels with minimal positive and maximal negative y coordinate
         -- these pixels correspond to the 'y' edges of the rectangle
    7.3. Select all empty pixels within the 'y' edges of the rectangle
    7.4. Find the pixels with minimal absolute x coordinate
         -- these pixels correspond to the 'x' edges of the rectangle
    8. Weight the excess charge within the rectangle additionally

    9. For each line calculate the energy of its interaction with the excess raster field

    Parameters
    ----------
    lines_batch : torch.Tensor
        of shape [batch_size, lines_n, params_n]

    rasters_batch : torch.Tensor
        of shape [batch_size, rasters_n]

    close_range_weight : number
        Should be twice the inverse of the lowest shading value.
    """
    batch_size, lines_n = lines_batch.shape[:2]
    x1 = lines_batch[..., 0]
    y1 = lines_batch[..., 1]
    x2 = lines_batch[..., 2]
    y2 = lines_batch[..., 3]
    half_width = lines_batch[..., 4] / 2
    lx = x2 - x1
    ly = y2 - y1
    length = torch.sqrt(lx ** 2 + ly ** 2)
    nonzero_length = torch.max(length, torch.full([1], division_epsilon, dtype=length.dtype, device=length.device))
    lx = lx / nonzero_length
    ly = ly / nonzero_length
    del nonzero_length

    with torch.no_grad():
        # 1. Render each line on binary supersample grid
        individual_rasterizations = render_lines(x1, y1, x2, y2, lines_batch[:, :, 4], sample_coordinates)

        # 2. Sum (OR) individual renderings -- this is total positive charge field
        patch_rasterizations = individual_rasterizations.sum(1, dtype=individual_rasterizations.dtype)

        # 3. For each line calculate the sum from step 2 minus the rendering of this line from step 1
        #    -- this is the excess positive charge field for this line
        others_rasterizations = individual_rasterizations ^ patch_rasterizations.unsqueeze(1)
        del patch_rasterizations, individual_rasterizations

        # 4. Subsample the renderings from step 3
        others_rasterizations = supersampling_strategy.subsample(others_rasterizations, dtype=dtype)

        # 5. Subtract the actual raster from the subsampled renderings from step 4
        #    -- this is the excess charge field for each line
        excess_raster = others_rasterizations
        rasters_batch = rasters_batch.type(dtype).reshape(batch_size, -1).unsqueeze(1)
        excess_raster -= rasters_batch
        del others_rasterizations

        # 6. For each line calculate coordinates of each excess charge in the coordinate system of this line,
        #    where the y axis is aligned along the length, the x axis is aligned along the width,
        #    and the origin is in the center of the line
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        del x1, x2, y1, y2
        cx, cy, lx, ly, excess_raster, rasters_batch, raster_x, raster_y = torch.broadcast_tensors(cx.unsqueeze(-1),
                                                                                                   cy.unsqueeze(-1),
                                                                                                   lx.unsqueeze(-1),
                                                                                                   ly.unsqueeze(-1),
                                                                                                   excess_raster,
                                                                                                   rasters_batch,
                                                                                                   raster_coordinates[
                                                                                                       0],
                                                                                                   raster_coordinates[
                                                                                                       1])

        # Translate points to canonical coordinate systems of the lines
        translated_raster_x = raster_x - cx
        translated_raster_y = raster_y - cy
        del raster_x, raster_y, cx, cy
        canonical_raster_x_abs = translated_raster_x * ly
        canonical_raster_x_abs -= translated_raster_y * lx
        canonical_raster_x_abs.abs_()
        canonical_raster_y = translated_raster_x * lx
        canonical_raster_y += translated_raster_y * ly
        del translated_raster_x, translated_raster_y, lx, ly

        # 7. For each line find the largest rectangle aligned along this line and filled with nonempty pixels only
        #    Such rectangle can be non unique, so define it like this:
        # 7.1. Select all empty pixels within `elementary_halfwidth` around the direction of the line, i.e |x| <= `elementary_halfwidth`
        # 7.2. Find the  pixels with minimal positive and maximal negative y coordinate
        #      -- these pixels correspond to the 'y' edges of the rectangle
        empty = rasters_batch <= empty_charge
        del rasters_batch
        candidates = canonical_raster_x_abs <= elementary_halfwidth
        candidates &= empty
        points_to_the_right = canonical_raster_y >= 0
        candidates_one_side = points_to_the_right & candidates
        assert candidates_one_side.any(dim=-1).all(), 'Couldn\'t find any empty pixel to the right'
        max_y = canonical_raster_y.masked_fill(~candidates_one_side, np.inf)
        max_y = max_y.min(dim=-1)[0]
        candidates_one_side = ~points_to_the_right
        del points_to_the_right
        candidates_one_side &= candidates
        assert candidates_one_side.any(dim=-1).all(), 'Couldn\'t find any empty pixel to the left'
        del candidates
        min_y = canonical_raster_y.masked_fill(~candidates_one_side, -np.inf)
        del candidates_one_side
        min_y = min_y.max(dim=-1)[0]

        # 7.3. Select all empty pixels within the 'y' edges of the rectangle
        within_y_edges = canonical_raster_y > min_y.unsqueeze(-1)
        within_y_edges &= canonical_raster_y < max_y.unsqueeze(-1)
        candidates = within_y_edges & empty
        del within_y_edges

        # 7.4. Find the pixels with minimal absolute x coordinate
        #      -- these pixels correspond to the 'x' edges of the rectangle
        max_x = canonical_raster_x_abs.masked_fill(~candidates, np.inf)
        max_x = max_x.min(dim=-1)[0]
        max_x.masked_fill_(~torch.isfinite(max_x), 0)

        # 8. Weight the excess charge within the rectangle additionally
        #    This is needed to avoid local minima of energy that are not optimal for the size energy
        visible_excess_charge = ~empty
        del empty
        visible_excess_charge &= canonical_raster_y <= (max_y.unsqueeze(-1) + visibility_padding)
        del max_y
        visible_excess_charge &= canonical_raster_y >= (min_y.unsqueeze(-1) - visibility_padding)
        del canonical_raster_y, min_y
        visible_excess_charge &= canonical_raster_x_abs <= (max_x.unsqueeze(-1) + visibility_padding)
        del canonical_raster_x_abs, max_x
        visible_excess_charge &= excess_raster < 0
        excess_raster = excess_raster.where(~visible_excess_charge, excess_raster * close_range_weight)
        del visible_excess_charge

    # 9. For each line calculate the energy of its interaction with the excess raster field
    mean_field_energy = line_to_point_energy(lines_batch, excess_raster).sum(-1).mean()
    del excess_raster
    return mean_field_energy


def mean_vector_field_energy_lines(lines_batch,supersampling_strategy = RegularSupersampling(4),division_epsilon = 1e-12):
    r"""...
    Algorithm is (for each batch):
    1. Render each line on binary supersample grid and subsample
    2. Put unit vector charge, aligned along the line, into every pixel shaded by the line
       -- this is its vector charge field
    3. Sum individual vector charge fields
    4. For each line calculate total complementary vector charge field
    5. For each line calculate the energy of its interaction with complementary vector charge field

    Parameters
    ----------
    lines_batch : torch.Tensor
        of shape [batch_size, lines_n, params_n]
    """
    batch_size, lines_n = lines_batch.shape[:2]

    with torch.no_grad():
        x1 = lines_batch[..., 0]
        y1 = lines_batch[..., 1]
        x2 = lines_batch[..., 2]
        y2 = lines_batch[..., 3]

        # 1. Render each line on binary supersample grid and subsample
        sample_coordinates = supersampling_strategy.supersample(raster_coordinates)
        individual_rasterizations = render_lines(x1, y1, x2, y2, lines_batch[:, :, 4], sample_coordinates)
        individual_rasterizations = supersampling_strategy.subsample(individual_rasterizations, dtype=dtype)

        # 2. Put unit vector charge, aligned along the line, into every pixel shaded by the line
        #    -- this is its vector charge field
        l_dir = torch.nn.functional.normalize(torch.stack([x2 - x1, y2 - y1]), dim=0, eps=division_epsilon)
        del x1, x2, y1, y2
        individual_vector_fields = l_dir.unsqueeze(-1) * individual_rasterizations
        del individual_rasterizations, l_dir

        # 3. Sum individual vector charge fields
        patch_vector_fields = individual_vector_fields.sum(2)

        # 4. For each line calculate total complementary vector charge field
        complimentary_vector_fields = patch_vector_fields.unsqueeze(2) - individual_vector_fields
        del patch_vector_fields, individual_vector_fields

    # 5. For each line calculate the energy of its interaction with complementary vector charge field
    mean_field_energy = line_to_vector_energy(lines_batch, complimentary_vector_fields).sum(-1).mean()
    del complimentary_vector_fields
    return mean_field_energy


def size_energy(lines_batch, rasters_batch, empty_charge=0, elementary_halfwidth=1 / 2,
                visibility_padding=2,division_epsilon = 1e-12):
    r"""...
    Algorithm is (for each batch):
    1. Render each line on binary supersample grid
    2. Sum (OR) individual renderings -- this is total positive charge field
    3. Subsample the total positive charge fields and individual rasterizations
    4. Subtract the actual raster from the total positive charge field from step 3
       to get the total excess raster for each line

    Steps 5-7 are needed to avoid local minima

    5. For each line calculate coordinates of each excess charge in the coordinate system of this line,
       where the y axis is aligned along the length, the x axis is aligned along the width,
       and the origin is in the center of the line
    6. For each line find the largest rectangle aligned along this line and filled with nonempty pixels only
       Such rectangle can be non unique, so define it like this:
    6.1. Select all empty pixels within elementary halfwidth around the direction of the line, i.e |x| <= elementary_halfwidth
         elementary halfwidth is e.g 1/2
    6.2. Find the  pixels with minimal positive and maximal negative y coordinate
         -- these pixels correspond to the 'y' edges of the rectangle
    6.3. Select all empty pixels within the 'y' edges of the rectangle
    6.4. Find the pixels with minimal absolute x coordinate
         -- these pixels correspond to the 'x' edges of the rectangle
    7. Limit the visible excess charge of this line to the pixels within its rectangle
       and add all pixels from its individual subsampled rendering
       -- this is the excess charge that the line interacts with

    8. For each line calculate the energy of its interaction with the excess raster field

    Parameters
    ----------
    lines_batch : torch.Tensor
        of shape [batch_size, lines_n, params_n]

    rasters_batch : torch.Tensor
        of shape [batch_size, rasters_n]
    """
    batch_size, lines_n = lines_batch.shape[:2]
    x1 = lines_batch[..., 0]
    y1 = lines_batch[..., 1]
    x2 = lines_batch[..., 2]
    y2 = lines_batch[..., 3]
    half_width = lines_batch[..., 4] / 2
    lx = x2 - x1
    ly = y2 - y1
    length = torch.sqrt(lx ** 2 + ly ** 2)
    nonzero_length = torch.max(length, torch.full([1], division_epsilon, dtype=length.dtype, device=length.device))
    lx = lx / nonzero_length
    ly = ly / nonzero_length
    del nonzero_length

    with torch.no_grad():
        # 1. Render each line on binary supersample grid
        individual_rasterizations = render_lines(x1, y1, x2, y2, lines_batch[:, :, 4], sample_coordinates)

        # 2. Sum (OR) individual renderings -- this is total positive charge field
        patch_rasterizations = individual_rasterizations.sum(1, dtype=individual_rasterizations.dtype)

        # 3. Subsample the total positive charge fields and individual rasterizations
        patch_rasterizations = supersampling_strategy.subsample(patch_rasterizations.unsqueeze(1))
        individual_rasterizations = supersampling_strategy.subsample(individual_rasterizations)

        # 4. Subtract the actual raster from the total positive charge field from step 3
        #    to get the total excess raster for each line
        excess_raster = patch_rasterizations
        rasters_batch = rasters_batch.type(dtype).reshape(batch_size, -1).unsqueeze(1)
        excess_raster -= rasters_batch
        del patch_rasterizations

        # 5. For each line calculate coordinates of each excess charge in the coordinate system of this line,
        #    where the y axis is aligned along the length, the x axis is aligned along the width,
        #    and the origin is in the center of the line
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        del x1, x2, y1, y2
        cx, cy, lx, ly, excess_raster, rasters_batch, raster_x, raster_y = torch.broadcast_tensors(cx.unsqueeze(-1),
                                                                                                   cy.unsqueeze(-1),
                                                                                                   lx.unsqueeze(-1),
                                                                                                   ly.unsqueeze(-1),
                                                                                                   excess_raster,
                                                                                                   rasters_batch,
                                                                                                   raster_coordinates[
                                                                                                       0],
                                                                                                   raster_coordinates[
                                                                                                       1])

        # Translate points to canonical coordinate systems of the lines
        translated_raster_x = raster_x - cx
        translated_raster_y = raster_y - cy
        del raster_x, raster_y, cx, cy
        canonical_raster_x_abs = translated_raster_x * ly
        canonical_raster_x_abs -= translated_raster_y * lx
        canonical_raster_x_abs.abs_()
        canonical_raster_y = translated_raster_x * lx
        canonical_raster_y += translated_raster_y * ly
        del translated_raster_x, translated_raster_y, lx, ly

        # 6. For each line find the largest rectangle aligned along this line and filled with nonempty pixels only
        #    Such rectangle can be non unique, so define it like this:
        # 6.1. Select all empty pixels within elementary halfwidth around the direction of the line, i.e |x| <= elementary_halfwidth
        #      elementary halfwidth is e.g 1/2
        # 6.2. Find the  pixels with minimal positive and maximal negative y coordinate
        #      -- these pixels correspond to the 'y' edges of the rectangle
        empty = rasters_batch <= empty_charge
        del rasters_batch
        candidates = canonical_raster_x_abs <= elementary_halfwidth
        candidates &= empty
        points_to_the_right = canonical_raster_y >= 0
        candidates_one_side = points_to_the_right & candidates
        assert candidates_one_side.any(dim=-1).all(), 'Couldn\'t find any empty pixel to the right'
        max_y = canonical_raster_y.masked_fill(~candidates_one_side, np.inf)
        max_y = max_y.min(dim=-1)[0]
        candidates_one_side = ~points_to_the_right
        del points_to_the_right
        candidates_one_side &= candidates
        assert candidates_one_side.any(dim=-1).all(), 'Couldn\'t find any empty pixel to the left'
        del candidates
        min_y = canonical_raster_y.masked_fill(~candidates_one_side, -np.inf)
        del candidates_one_side
        min_y = min_y.max(dim=-1)[0]

        # 6.3. Select all empty pixels within the 'y' edges of the rectangle
        within_y_edges = canonical_raster_y > min_y.unsqueeze(-1)
        within_y_edges &= canonical_raster_y < max_y.unsqueeze(-1)
        candidates = within_y_edges & empty
        del within_y_edges

        # 6.4. Find the pixels with minimal absolute x coordinate
        #      -- these pixels correspond to the 'x' edges of the rectangle
        max_x = canonical_raster_x_abs.masked_fill(~candidates, np.inf)
        max_x = max_x.min(dim=-1)[0]
        max_x.masked_fill_(~torch.isfinite(max_x), 0)

        # 7. Limit the visible excess charge of this line to the pixels within its rectangle
        #    and add all pixels from its individual subsampled rendering
        #    -- this is the excess charge that the line interacts with
        visible_excess_charge = ~empty
        del empty
        visible_excess_charge &= canonical_raster_y <= (max_y.unsqueeze(-1) + visibility_padding)
        del max_y
        visible_excess_charge &= canonical_raster_y >= (min_y.unsqueeze(-1) - visibility_padding)
        del canonical_raster_y, min_y
        visible_excess_charge &= canonical_raster_x_abs <= (max_x.unsqueeze(-1) + visibility_padding)
        del canonical_raster_x_abs, max_x
        excess_raster = excess_raster.where(visible_excess_charge, individual_rasterizations)
        del visible_excess_charge, individual_rasterizations
        ### ax_debug.imshow(excess_raster[0, 0].detach().cpu().reshape(padded_h, padded_w))

    # 8. For each line calculate the energy of its interaction with the excess raster field
    mean_field_energy = line_to_point_energy(lines_batch, excess_raster).sum(-1).mean()
    del excess_raster
    return mean_field_energy


def reinit_excess_lines(cx, cy, width, length, excess_raster, patches_to_consider, min_raster_to_fill=.5,
                        min_width=1 / 8, initial_width=1, initial_length=1):
    r"""
    Algorithm is:
    1. In each patch find the maximum value of excess raster
       that is not already covered by some vector primitive
    2. Find the patches for which this value is larger than `min_raster_to_fill`
       and that are from `patches_to_consider`
       -- only these patches need reinitialization
    3. In every such patch find the line with the minimal width
    4. Find the patches for which this value is lower than `min_width`
         (the patch has invisible lines that are not already 'working')
       and that need reinitialization
       -- only these patches will be reinitialized
    5. In every such patch put the line with the minimal width from step 3
       into the position maximum excess raster from step 1
       and reinitialize the length and width of this line
    """
    with torch.no_grad():
        # 1. In each patch find the maximum value of excess raster
        #    that is not already covered by some vector primitive
        max_excess, max_excess_id = torch.max(excess_raster, dim=-1)
        # 2. Find the patches for which this value is larger than `min_raster_to_fill`
        #    and that are from `patches_to_consider`
        #    -- only these patches need reinitialization
        patches_to_reinit = torch.tensor(patches_to_consider, dtype=torch.bool, device=cx.device)
        patches_to_consider_to_reinit = max_excess >= min_raster_to_fill
        patches_to_reinit.masked_scatter_(patches_to_reinit, patches_to_consider_to_reinit)
        if patches_to_reinit.sum() == 0:
            return
        # 3. In every such patch find the line with the minimal width
        min_lines_width, lines_to_reinit = torch.min(width[patches_to_reinit], dim=-1)
        # 4. Find the patches for which this value is lower than `min_width`
        #      (the patch has invisible lines that are not already 'working')
        #    and that need reinitialization
        #    -- only these patches will be reinitialized
        patches_to_reinit_with_excess_lines = min_lines_width < min_width
        patches_to_consider_to_reinit[patches_to_consider_to_reinit] &= patches_to_reinit_with_excess_lines
        patches_to_reinit[patches_to_reinit] &= patches_to_reinit_with_excess_lines
        if patches_to_reinit.sum() == 0:
            return
        lines_to_reinit = lines_to_reinit[patches_to_reinit_with_excess_lines]
        # 5. In every such patch put the line with the minimal width from step 3
        #    into the position maximum excess raster from step 1
        #    and reinitialize the length and width of this line
        cx.data[patches_to_reinit, lines_to_reinit] = raster_coordinates[
            0, max_excess_id[patches_to_consider_to_reinit]]
        cy.data[patches_to_reinit, lines_to_reinit] = raster_coordinates[
            1, max_excess_id[patches_to_consider_to_reinit]]
        length.data[patches_to_reinit, lines_to_reinit] = initial_length
        width.data[patches_to_reinit, lines_to_reinit] = initial_width


def snap_lines(cx, cy, theta, length, width, pos_optimizer, size_optimizer, width_threshold=1 / 4, coord_threshold=1.5,
               direction_threshold=5, min_linear=2**-8):
    r"""
    Algorith is for each 'this' line:
    1.C. Select the collinear other lines
    1.C.21. Among them select the lines with p2 == this p1
    1.C.21.W. Among them select the lines with same widths
              With the other parts of the algorithm, chances are low that such line is nonunique for 'this' line
    2.C.21. Snap the other line with 'this' line, collapse 'this' line,
            and keep track of the modified and collapsed lines
     Do the same for other variants of 1-2
    3. Prevent rocking after snaps
    4. Prevent size freezing after snaps
    5. Reset collapsed lines
    """
    modified_lines = torch.full(cx.shape, False, dtype=torch.bool, device=cx.device)
    collapsed_lines = torch.full(cx.shape, False, dtype=torch.bool, device=cx.device)
    with torch.no_grad():
        # For each 'this' line
        for line_i in range(cx.shape[1] - 1):
            length_others = length[:, line_i + 1:]
            width_others = width[:, line_i + 1:]
            theta_others = theta[:, line_i + 1:]
            cos_others = torch.cos(theta_others)
            sin_others = torch.sin(theta_others)
            cx_others = cx[:, line_i + 1:]
            cy_others = cy[:, line_i + 1:]
            x1_others = cx_others - length_others * cos_others / 2
            y1_others = cy_others - length_others * sin_others / 2
            x2_others = cx_others + length_others * cos_others / 2
            y2_others = cy_others + length_others * sin_others / 2

            length_this = length[:, line_i:line_i + 1].expand_as(length_others)
            width_this = width[:, line_i:line_i + 1].expand_as(width_others)
            theta_this = theta[:, line_i:line_i + 1].expand_as(theta_others)
            cos_this = torch.cos(theta_this)
            sin_this = torch.sin(theta_this)
            cx_this = cx[:, line_i:line_i + 1].expand_as(cx_others)
            cy_this = cy[:, line_i:line_i + 1].expand_as(cy_others)
            x1_this = cx_this - length_this * cos_this / 2
            y1_this = cy_this - length_this * sin_this / 2
            x2_this = cx_this + length_this * cos_this / 2
            y2_this = cy_this + length_this * sin_this / 2

            cos_theta_dif = torch.cos(theta_others - theta_this)
            cos_threshold = np.cos(direction_threshold * np.pi / 180)

            not_modified_this = torch.full(length_others.shape, True, dtype=torch.bool, device=length_others.device)

            # 1.C. Select the collinear other lines
            close_directions = cos_theta_dif > cos_threshold

            # 1.C.21. Among them select the lines with p2 == this p1
            close_other2_this1 = close_directions.clone()
            close_other2_this1[close_directions] &= torch.abs(
                x2_others[close_directions] - x1_this[close_directions]) <= coord_threshold
            close_other2_this1[close_directions] &= torch.abs(
                y2_others[close_directions] - y1_this[close_directions]) <= coord_threshold

            # 1.C.21.W. Among them select the lines with same widths
            #  With the other parts of the algorithm, chances are low that such line is nonunique for 'this' line
            close_width = close_other2_this1.clone()
            close_width[close_other2_this1] &= torch.abs(
                width_others[close_other2_this1] - width_this[close_other2_this1]) < width_threshold

            # 2.C.21. Snap the other line with 'this' line,
            new_x2_others = x2_this[close_width]
            new_y2_others = y2_this[close_width]
            cx_others.data[close_width] = (new_x2_others + x1_others[close_width]) / 2
            cy_others.data[close_width] = (new_y2_others + y1_others[close_width]) / 2
            length_others.data[close_width] = torch.sqrt(
                (new_x2_others - x1_others[close_width]) ** 2 + (new_y2_others - y1_others[close_width]) ** 2)
            #         collapse 'this' line
            length_this.data[close_width] = min_linear
            width_this.data[close_width] = min_linear
            #         and keep track of the modified and collapsed lines
            modified_lines[:, line_i + 1:][close_width] = True
            collapsed_lines[:, line_i][close_width.max(dim=-1)[0]] = True
            not_modified_this[close_width.max(dim=-1)[0]] = False
            close_directions[close_width.max(dim=-1)[0]] = False
            del new_x2_others, new_y2_others, close_other2_this1, close_width

            # 1.C.12. other's p1 = this p2
            close_other1_this2 = close_directions.clone()
            close_other1_this2[close_directions] &= torch.abs(
                x1_others[close_directions] - x2_this[close_directions]) <= coord_threshold
            close_other1_this2[close_directions] &= torch.abs(
                y1_others[close_directions] - y2_this[close_directions]) <= coord_threshold

            # 1.C.12.W. same widths
            close_width = close_other1_this2.clone()
            close_width[close_other1_this2] &= torch.abs(
                width_others[close_other1_this2] - width_this[close_other1_this2]) < width_threshold

            # 2.C.12. Snap the other line with 'this' line,
            new_x1_others = x1_this[close_width]
            new_y1_others = y1_this[close_width]
            cx_others.data[close_width] = (new_x1_others + x2_others[close_width]) / 2
            cy_others.data[close_width] = (new_y1_others + y2_others[close_width]) / 2
            #         collapse 'this' line
            length_others.data[close_width] = torch.sqrt(
                (x2_others[close_width] - new_x1_others) ** 2 + (y2_others[close_width] - new_y1_others) ** 2)
            length_this.data[close_width] = min_linear
            width_this.data[close_width] = min_linear
            #         and keep track of the modified and collapsed lines
            modified_lines[:, line_i + 1:][close_width] = True
            collapsed_lines[:, line_i][close_width.max(dim=-1)[0]] = True
            not_modified_this[close_width.max(dim=-1)[0]] = False
            del new_x1_others, new_y1_others, close_other1_this2, close_width

            # 1.A. anticollinear
            close_directions = (cos_theta_dif < -cos_threshold) & not_modified_this

            # 1.A.22. other's p2 = this p2
            close_other2_this2 = close_directions.clone()
            close_other2_this2[close_directions] &= torch.abs(
                x2_others[close_directions] - x2_this[close_directions]) <= coord_threshold
            close_other2_this2[close_directions] &= torch.abs(
                y2_others[close_directions] - y2_this[close_directions]) <= coord_threshold

            # 1.A.22.W. same widths
            close_width = close_other2_this2.clone()
            close_width[close_other2_this2] &= torch.abs(
                width_others[close_other2_this2] - width_this[close_other2_this2]) < width_threshold

            # 2.A.22. Snap the other line with 'this' line,
            new_x2_others = x1_this[close_width]
            new_y2_others = y1_this[close_width]
            cx_others.data[close_width] = (new_x2_others + x1_others[close_width]) / 2
            cy_others.data[close_width] = (new_y2_others + y1_others[close_width]) / 2
            length_others.data[close_width] = torch.sqrt(
                (new_x2_others - x1_others[close_width]) ** 2 + (new_y2_others - y1_others[close_width]) ** 2)
            #         collapse 'this' line
            length_this.data[close_width] = min_linear
            width_this.data[close_width] = min_linear
            #         and keep track of the modified and collapsed lines
            modified_lines[:, line_i + 1:][close_width] = True
            collapsed_lines[:, line_i][close_width.max(dim=-1)[0]] = True
            not_modified_this[close_width.max(dim=-1)[0]] = False
            close_directions[close_width.max(dim=-1)[0]] = False
            del new_x2_others, new_y2_others, close_other2_this2, close_width

            # 1.A.11. other's p1 = this p1
            close_other1_this1 = close_directions.clone()
            close_other1_this1[close_directions] &= torch.abs(
                x1_others[close_directions] - x1_this[close_directions]) <= coord_threshold
            close_other1_this1[close_directions] &= torch.abs(
                y1_others[close_directions] - y1_this[close_directions]) <= coord_threshold

            # 1.A.11.W. same widths
            close_width = close_other1_this1.clone()
            close_width[close_other1_this1] &= torch.abs(
                width_others[close_other1_this1] - width_this[close_other1_this1]) < width_threshold

            # 2.A.11. Snap the other line with 'this' line,
            new_x1_others = x2_this[close_width]
            new_y1_others = y2_this[close_width]
            cx_others.data[close_width] = (new_x1_others + x2_others[close_width]) / 2
            cy_others.data[close_width] = (new_y1_others + y2_others[close_width]) / 2
            length_others.data[close_width] = torch.sqrt(
                (x2_others[close_width] - new_x1_others) ** 2 + (y2_others[close_width] - new_y1_others) ** 2)
            #         collapse 'this' line
            length_this.data[close_width] = min_linear
            width_this.data[close_width] = min_linear
            #         and keep track of the modified and collapsed lines
            modified_lines[:, line_i + 1:][close_width] = True
            collapsed_lines[:, line_i][close_width.max(dim=-1)[0]] = True
            del new_x1_others, new_y1_others, close_other1_this1, close_width, close_directions

    # 3. Prevent rocking after snaps
    angle_damper = pos_optimizer.state[theta]['exp_avg_sq'].new_full([1], 1e6)
    pos_optimizer.state[theta]['exp_avg'].data[modified_lines] = 0
    pos_optimizer.state[theta]['exp_avg_sq'].data[modified_lines] = pos_optimizer.state[theta]['exp_avg_sq'].data[
        modified_lines].max(angle_damper)
    pos_optimizer.state[cx]['exp_avg'].data[modified_lines] = 0
    pos_optimizer.state[cx]['exp_avg_sq'].data[modified_lines] = 0
    pos_optimizer.state[cy]['exp_avg'].data[modified_lines] = 0
    pos_optimizer.state[cy]['exp_avg_sq'].data[modified_lines] = 0
    # 4. Prevent size freezing after snaps
    size_optimizer.state[length]['exp_avg'].data[modified_lines] = 0
    size_optimizer.state[length]['exp_avg_sq'].data[modified_lines] = 0
    size_optimizer.state[width]['exp_avg'].data[modified_lines] = 0
    size_optimizer.state[width]['exp_avg_sq'].data[modified_lines] = 0
    # 5. Reset collapsed lines
    pos_optimizer.state[cx]['exp_avg'].data[collapsed_lines] = 0
    pos_optimizer.state[cx]['exp_avg_sq'].data[collapsed_lines] = 0
    pos_optimizer.state[cy]['exp_avg'].data[collapsed_lines] = 0
    pos_optimizer.state[cy]['exp_avg_sq'].data[collapsed_lines] = 0
    pos_optimizer.state[theta]['exp_avg'].data[collapsed_lines] = 0
    pos_optimizer.state[theta]['exp_avg_sq'].data[collapsed_lines] = 0
    size_optimizer.state[length]['exp_avg'].data[collapsed_lines] = 0
    size_optimizer.state[length]['exp_avg_sq'].data[collapsed_lines] = 0
    size_optimizer.state[width]['exp_avg'].data[collapsed_lines] = 0
    size_optimizer.state[width]['exp_avg_sq'].data[collapsed_lines] = 0


def collapse_redundant_lines(cx, cy, theta, length, width, patches_to_consider, enum_type=torch.int8,
                             min_linear=2**-8):
    r"""
    Algorithm is:
    1. Render each line on binary supersample grid
    2. Sum (OR) individual renderings
    3. For each line check if the line is redundant:
       Remember, that our renderings on the subsample grid are binary.
       For each pixel where the current line is rendered we subtract 2 from the total number of lines in this pixel.
       If this line is the only one in this pixel, then the result is negative;
        if this line is not the only one, then the result is nonnegative.
       The line is redundant iff there are no pixels where it is not the only one
    4. Remove redundant lines from the total rendering
    """
    # assert lines_n < 128, 'Risk of overflow! Change int8 to something else'
    lines_n = cx.shape[1]
    with torch.no_grad():
        x2 = cx.data[patches_to_consider] + length.data[patches_to_consider] * torch.cos(
            theta.data[patches_to_consider]) / 2
        x1 = cx.data[patches_to_consider] - length.data[patches_to_consider] * torch.cos(
            theta.data[patches_to_consider]) / 2
        y2 = cy.data[patches_to_consider] + length.data[patches_to_consider] * torch.sin(
            theta.data[patches_to_consider]) / 2
        y1 = cy.data[patches_to_consider] - length.data[patches_to_consider] * torch.sin(
            theta.data[patches_to_consider]) / 2

        # 1. Render each line on binary supersample grid
        individual_rasterizations = render_lines(x1, y1, x2, y2, width[patches_to_consider], sample_coordinates).type(
            enum_type)

        # 2. Sum (OR) individual renderings
        patch_rasterizations = individual_rasterizations.sum(1, dtype=enum_type)

        for line_i in range(lines_n):
            patches_to_work_with = torch.tensor(patches_to_consider, dtype=torch.bool, device=cx.device)
            # 3. Check if a line is redundant.
            # Remember, that our renderings on the subsample grid are binary.
            # For each pixel where the current line is rendered
            # we subtract 2 from the total number of lines in this pixel.
            # If this line is the only one in this pixel, then the result is negative;
            # if this line is not the only one, then the result is nonnegative.
            patch_rasterizations_with_negative_line = individual_rasterizations[:, line_i].clone()
            patch_rasterizations_with_negative_line *= -2
            patch_rasterizations_with_negative_line += patch_rasterizations

            # The line is redundant iff there are no pixels where it is not the only one
            line_is_redundant = (patch_rasterizations_with_negative_line >= 0).all(dim=-1)
            del patch_rasterizations_with_negative_line

            # 4. Remove redundant lines from the total rendering
            patch_rasterizations[line_is_redundant] -= individual_rasterizations[line_is_redundant, line_i]
            patches_to_work_with[patches_to_work_with] = line_is_redundant
            width[patches_to_work_with, line_i] = min_linear
            length[patches_to_work_with, line_i] = min_linear
        del individual_rasterizations, patch_rasterizations, line_is_redundant


def constrain_parameters(cx, cy, theta, length, width, canvas_width, canvas_height, size_optimizer, padding= 3 - 2,
                         min_linear=2**-8, dwarfness_ratio=1,division_epsilon = 1e-12):
    r"""
    1. Constrain width and length to be non less than `min_linear` which is nonzero
       to prevent 'dying' of the lines (any position of a zero-sized line is optimal)
    3. Swap width and length for short and wide 'dwarf' lines
    4. Limit positions of the lines to the canvas
       Nonzero padding is used to prevent nonstability for the lines trying to fit super-narrow raster
       (i.e, with small shading value) at the very edge of the canvas
       We use the `size_energy` padding value minus 2
       since `size_energy` needs the line to be at least 2 pixels away from the edge
    5. Limit the length of the line w.r.t the edges of the canvas
    """
    ## # 0. Reset exponential averages for collapsed lines
    ## size_optimizer.state[length]['exp_avg'].data[length <= min_linear] = 0
    ## size_optimizer.state[length]['exp_avg_sq'].data[length <= min_linear] = 0
    ## size_optimizer.state[width]['exp_avg'].data[width <= min_linear] = 0
    ## size_optimizer.state[width]['exp_avg_sq'].data[width <= min_linear] = 0

    # 1. Constrain width and length to be non less than `min_linear` which is nonzero
    #    to prevent 'dying' of the lines (any position of a zero-sized line is optimal)
    width.data.clamp_(min=min_linear)
    length.data.clamp_(min=min_linear)
    # 2. Keep theta in [0, 2pi)
    theta.data.remainder_(np.pi * 2)

    # 3. Swap width and length for short and wide 'dwarf' lines
    dwarf_lines = width.data > length.data * dwarfness_ratio
    width.data[dwarf_lines], length.data[dwarf_lines] = length.data[dwarf_lines], width.data[dwarf_lines]
    theta.data[dwarf_lines] += np.pi / 2
    size_optimizer.state[length]['exp_avg'].data[dwarf_lines], size_optimizer.state[width]['exp_avg'].data[
        dwarf_lines] = size_optimizer.state[width]['exp_avg'].data[dwarf_lines], \
                       size_optimizer.state[length]['exp_avg'].data[dwarf_lines]
    size_optimizer.state[length]['exp_avg_sq'].data[dwarf_lines], size_optimizer.state[width]['exp_avg_sq'].data[
        dwarf_lines] = size_optimizer.state[width]['exp_avg_sq'].data[dwarf_lines], \
                       size_optimizer.state[length]['exp_avg_sq'].data[dwarf_lines]

    # 4. Limit positions of the lines to the canvas
    #    Nonzero padding is used to prevent nonstability for the lines trying to fit super-narrow raster
    #    (i.e, with small shading value) at the very edge of the canvas
    #    We use the `size_energy` padding value minus 2
    #    since `size_energy` needs the line to be at least 2 pixels away from the edge
    cx.data.clamp_(min=-padding, max=canvas_width + padding)
    cy.data.clamp_(min=-padding, max=canvas_height + padding)

    # 5. Limit the length of the line w.r.t the edges of the canvas
    epsiloned_2cos = torch.abs(torch.cos(theta.data)) / 2 + division_epsilon
    epsiloned_2sin = torch.abs(torch.sin(theta.data)) / 2 + division_epsilon
    maxl = torch.min(torch.stack([
        (cx.data + padding) / epsiloned_2cos, (canvas_width + padding - cx.data) / epsiloned_2cos,
        (cy.data + padding) / epsiloned_2sin, (canvas_height + padding - cy.data) / epsiloned_2sin
    ], -1), dim=-1)[0]
    length.data = torch.min(length.data, maxl).clamp(min=min_linear)


def my_iou_score(vector_rendering, rasters_batch, average=None):
    _vector = ((1 - vector_rendering.detach().cpu()).clamp(0, 1) * 255).type(torch.uint8).numpy()
    _raster = ((1 - rasters_batch.detach().cpu()).clamp(0, 1) * 255).type(torch.uint8).numpy()
    return iou_score(_raster, _vector, binarization='median', average=average)