import numpy as np
import torch

from util_files.rendering.cairo import render_with_skeleton, PT_LINE

from ..primitive_tensor import PrimitiveTensor
from ...parameters import coordinates_constrain_padding, division_epsilon, dwarfness_ratio, refinement_linecaps,\
    refinement_linejoin, min_linear_size, visibility_width_threshold

from .procedures import calculate_canonical_coordinates


class LineTensor(PrimitiveTensor):
    def __init__(self, p1, p2, width, dtype=None, device=None):
        r"""

        Parameters
        ----------
        p1 : array_like
            of shape [patches_n, spatial_dims_n, primitives_n]
        p2 : array_like
            of shape [patches_n, spatial_dims_n, primitives_n]
        width : array_like
            of shape [patches_n, 1, primitives_n]
        dtype : torch.dtype, optional
            The desired data type of the parameters and all computations.
            If None, uses a global default see ``torch.set_default_tensor_type()``
        device : torch.device, optional
            The desired device of the raster grid and all computations. If None, uses the current device for `dtype`
        """
        super().__init__(dtype=dtype, device=device)
        device = self.device
        dtype = self.dtype

        p1 = torch.as_tensor(p1, dtype=dtype)
        p2 = torch.as_tensor(p2, dtype=dtype)
        width = torch.as_tensor(width, dtype=dtype)
        assert p1.shape == p2.shape and (width.shape[0] == p1.shape[0]) and (width.shape[2] == p1.shape[2])

        self.patches_n, _, self.primitives_n = p1.shape

        # calculate parameters
        self._width = width

        self._midpoint = ((p1 + p2) / 2)
        dx = p2[:, 0] - p1[:, 0]
        dy = p2[:, 1] - p1[:, 1]
        self._length = torch.sqrt(dx ** 2 + dy ** 2).unsqueeze(1)
        self._theta = torch.atan2(dy, dx).unsqueeze(1)

        # intermediate parameters
        self._p1 = None
        self._p2 = None
        self._dir = None
        self._unit_energy = None
        self._pixel_coords = None

        # optimized parameters
        self._midpoint = self._midpoint.to(device).requires_grad_()
        self._theta = self._theta.to(device).requires_grad_()
        self._length = self._length.to(device).requires_grad_()
        self._width = self._width.to(device).requires_grad_()
        self.canonical_parameters = [{'parameter': self._midpoint, 'lr_factor': 1},
                                     {'parameter': self._theta, 'lr_factor': 1e-2},
                                     {'parameter': self._length, 'lr_factor': 1},
                                     {'parameter': self._width, 'lr_factor': 1e-1}]

    @property
    def dir(self):
        if self._dir is None:
            self._dir = torch.cat([torch.cos(self.theta), torch.sin(self.theta)], dim=1)
        return self._dir

    @property
    def length(self):
        if self.size_fixed:
            return self._length.data
        else:
            return self._length

    @property
    def midpoint(self):
        if self.pos_fixed:
            return self._midpoint.data
        else:
            return self._midpoint

    @property
    def p1(self):
        if self._p1 is None:
            self._p1 = self.midpoint - self.length * self.dir / 2
        return self._p1

    @property
    def p2(self):
        if self._p2 is None:
            self._p2 = self.midpoint + self.length * self.dir / 2
        return self._p2

    @property
    def theta(self):
        if self.pos_fixed:
            return self._theta.data
        else:
            return self._theta

    @property
    def width(self):
        if self.size_fixed:
            return self._width.data
        else:
            return self._width

    def calculate_canonical_coordinates(self, pixel_coords, from_midpoint=False):
        r"""

        Parameters
        ----------
        pixel_coords : torch.Tensor
            of shape [spatial_dims_n, pixels_n]

        Returns
        -------
        canonical_x : torch.Tensor
            of shape [patches_n, primitives_n, pixels_n]
        canonical_y : torch.Tensor
            of shape [patches_n, primitives_n, pixels_n]
        """
        canonical_x, canonical_y = calculate_canonical_coordinates(pixel_coords, self.p1, self.dir)
        if from_midpoint:
            patches_n, primitives_n = canonical_y.shape[:2]
            canonical_y = canonical_y - self._length.data.reshape(patches_n, primitives_n, 1) / 2
        return canonical_x, canonical_y

    def constrain_parameters(self, patch_width, patch_height, division_epsilon=division_epsilon):
        # 1. Constrain width and length to be non less than `min_linear` which is nonzero
        #    to prevent 'dying' of the lines (any position of a zero-sized line is optimal)
        self._width.data.clamp_(min=min_linear_size)
        self._length.data.clamp_(min=min_linear_size)

        # 2. Keep theta in [0, 2pi)
        t = self._theta.data
        t.remainder_(np.pi * 2)

        # 3. Swap width and length for short and wide 'dwarf' lines
        w = self._width.data
        l = self._length.data
        dwarf_lines = w > l * dwarfness_ratio
        w[dwarf_lines], l[dwarf_lines] = l[dwarf_lines], w[dwarf_lines]
        t[dwarf_lines] += np.pi / 2

        # 3.1. Swap moments etc for dwarf lines
        optimizer = self.optimizer
        w = self._width
        l = self._length
        if isinstance(optimizer, torch.optim.Adam) and 'exp_avg' in optimizer.state[l]:
            optimizer.state[l]['exp_avg'].data[dwarf_lines], optimizer.state[w]['exp_avg'].data[dwarf_lines] = \
                optimizer.state[w]['exp_avg'].data[dwarf_lines], optimizer.state[l]['exp_avg'].data[dwarf_lines]
            optimizer.state[l]['exp_avg_sq'].data[dwarf_lines], optimizer.state[w]['exp_avg_sq'].data[dwarf_lines] = \
                optimizer.state[w]['exp_avg_sq'].data[dwarf_lines], optimizer.state[l]['exp_avg_sq'].data[dwarf_lines]

        # 4. Limit positions of the lines to the canvas
        #    Nonzero padding is used to prevent nonstability for the lines trying to fit super-narrow raster
        #    (i.e, with small shading value) at the very edge of the canvas
        self._midpoint.data[:, 0].clamp_(min=-coordinates_constrain_padding,
                                         max=patch_width + coordinates_constrain_padding)
        self._midpoint.data[:, 1].clamp_(min=-coordinates_constrain_padding,
                                         max=patch_height + coordinates_constrain_padding)

        # 5. Limit the length of the line w.r.t the edges of the canvas
        m = self._midpoint.data
        l = self._length.data
        epsiloned_2cos = t[:, 0].cos().abs() / 2 + division_epsilon
        epsiloned_2sin = t[:, 0].sin().abs() / 2 + division_epsilon
        maxl = torch.min(torch.stack([
            (m[:, 0] + coordinates_constrain_padding) / epsiloned_2cos,
            (-m[:, 0] + patch_width + coordinates_constrain_padding) / epsiloned_2cos,
            (m[:, 1] + coordinates_constrain_padding) / epsiloned_2sin,
            (-m[:, 1] + patch_height + coordinates_constrain_padding) / epsiloned_2sin
        ], dim=0), dim=0)[0]
        constrained_l = torch.min(l, maxl.unsqueeze(1)).clamp(min=min_linear_size)
        l.copy_(constrained_l)

    def get_vector_field_at(self, t):
        r"""

        Parameters
        ----------
        t : torch.Tensor
            of shape [patches_n, primitives_n, pixels_n]

        Returns
        -------
        """
        patches_n, primitives_n, pixels_n = t.shape
        spatial_dims_n = 2

        vector_field = self.dir.detach().clone()
        vector_field = vector_field.reshape(patches_n, spatial_dims_n, primitives_n, 1)\
                                   .expand(patches_n, spatial_dims_n, primitives_n, pixels_n)
        return vector_field


    def invalidate_pos_dependent(self):
        self._p1 = None  # depends on midpoint and theta
        self._p2 = None  # depends on midpoint and theta
        self._dir = None  # depends on theta
        self._unit_energy = None  # depends on midpoint and theta

    def invalidate_size_dependent(self):
        self._p1 = None  # depends on length
        self._p2 = None  # depends on length
        self._unit_energy = None  # depends on length and width

    def reinit_primitives(self, primitives_to_reinit, coords_to_reinit_at, initial_width, initial_length):
        r"""

        Parameters
        ----------
        primitives_to_reinit : torch.Tensor
            of shape [patches_n, primitives_n]
        coords_to_reinit_at : torch.Tensor
            of shape [patches_n, spatial_dims_n]
        """
        midpoint = self._midpoint.data

        # 1. Put midpoint at coords_to_reinit
        patches_n, spatial_dims_n, primitives_n = midpoint.shape
        primitives_to_reinit = primitives_to_reinit.reshape(patches_n, 1, primitives_n)
        midpoint.masked_scatter_(primitives_to_reinit, coords_to_reinit_at)

        # 2. Reinit lengths and widths
        self._length.data.masked_fill_(primitives_to_reinit, initial_length)
        self._width.data.masked_fill_(primitives_to_reinit, initial_width)

        self.invalidate_pos_dependent()
        self.invalidate_size_dependent()

    def render_single_primitive_with_cairo(self, ctx, patch_i, primitive_i, min_width=0):
        width = self.width[patch_i, 0, primitive_i].detach().cpu().numpy()
        if width < min_width:
            return

        ctx.move_to(*self.p1[patch_i, :, primitive_i].detach().cpu().numpy())
        ctx.line_to(*self.p2[patch_i, :, primitive_i].detach().cpu().numpy())
        ctx.set_line_width(width)
        ctx.stroke()

    def render_skeleton_total(self, width, height, visibility_width_threshold=visibility_width_threshold, scaling=4,
                              line_color=(31 / 255, 119 / 255, 180 / 255), line_width=2, control_line_width=1,
                              node_color=(255 / 255, 127 / 255, 14 / 255), node_size=4, control_node_size=2):
        r"""

        Parameters
        ----------
        width : int
            width of the patch
        height : int
            height of the patch

        Returns
        -------
        renderings : np.ndarray
            of shape [patches_n, height, width]
        """
        widths = self.width.detach().cpu().numpy()
        visible = widths[:, 0] >= visibility_width_threshold
        widths = widths * 0
        pars = np.concatenate([self.p1.detach().cpu().numpy(),
                               self.p2.detach().cpu().numpy(), widths], 1).swapaxes(1, 2)
        pars *= scaling
        return np.stack([render_with_skeleton(
            {PT_LINE: patch[visibility]}, [width * scaling, height * scaling], data_representation='vahe',
            linecaps=refinement_linecaps, linejoin=refinement_linejoin, line_color=line_color, line_width=line_width,
            node_color=node_color, node_size=node_size, control_line_width=control_line_width,
            control_node_size=control_node_size)
            for patch, visibility in zip(pars, visible)])

    def unit_energy(self, pixel_coords):
        r"""

        Parameters
        ----------
        pixel_coords : torch.Tensor
            of shape [spatial_dims_n, pixels_n].

        Returns
        -------
        energies : torch.Tensor
            of shape [patches_n, primitives_n, pixels_n]
        """
        # reuse cached values if they weren't invalidated and pixel_coords are the same
        if ((self._unit_energy is not None) and (self._pixel_coords is not None) and
                torch.all(self._pixel_coords == pixel_coords)):
            return self._unit_energy

        # calculate energy and cache the values
        canonical_pixel_x, canonical_pixel_y = self.calculate_canonical_coordinates(pixel_coords)
        self._unit_energy = self.energy_procedures.unit_energy_line_to_canonical_point(
            self.width[:, 0].unsqueeze(-1) / 2, self.length[:, 0].unsqueeze(-1), canonical_pixel_x, canonical_pixel_y)
        del canonical_pixel_x, canonical_pixel_y
        self._pixel_coords = pixel_coords

        return self._unit_energy
