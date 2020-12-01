import numpy as np
import torch
from torch import nn

from util_files.rendering.cairo import render_with_skeleton, PT_QBEZIER_B

from ..primitive_tensor import PrimitiveTensor
from ...parameters import coordinates_constrain_padding, division_epsilon, dwarfness_ratio, elementary_halfwidth,\
    empty_pixel_tolerance, min_linear_size, neighbourhood_padding, qbezier_max_dt_end,\
    qbezier_min_fold_halfangle_radians, qbezier_y_neighbourhood_padding, refinement_linecaps, refinement_linejoin,\
    reinit_initial_length, reinit_initial_width, visibility_width_threshold
from util_files.simplification.join_qb import join_quad_beziers


class QuadraticBezierTensor(PrimitiveTensor):
    def __init__(self, p1, p2, p3, width, dtype=None, device=None):
        r"""

        Parameters
        ----------
        p1 : array_like
            of shape [patches_n, spatial_dims_n, primitives_n]
        p2 : array_like
            of shape [patches_n, spatial_dims_n, primitives_n]
        p3 : array_like
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
        p3 = torch.as_tensor(p3, dtype=dtype)
        width = torch.as_tensor(width, dtype=dtype)
        assert ((p1.shape == p2.shape) and (p1.shape == p3.shape) and
                (width.shape[0] == p1.shape[0]) and (width.shape[2] == p1.shape[2]))

        patches_n, spatial_dims_n, primitives_n = p1.shape
        self.patches_n = patches_n
        self.primitives_n = primitives_n

        # intermediate parameters
        self._p1 = None
        self._p2 = None
        self._p3 = None
        self._b_to_p1 = None
        self._b_to_p3 = None

        self._segments_p1 = None
        self._segments_dir = None
        self._segments_length = None

        self._canonical_pixel_coords = None
        self._pixel_coords = None
        self._in_neighbourhood = None

        # optimized parameters
        _ = self._width = width.to(device).requires_grad_()
        #TODO for Torch 1.4+ new_empty not work with requires_grad
        # self._p2_to_p1_len = _.new_empty([patches_n, 1, primitives_n], requires_grad=True)
        # self._p2_to_p1_len.requires_grad(True)
        # empty with type, size and device
        self._p2_to_p1_len = _.new_empty([patches_n, 1, primitives_n], requires_grad=True)
        self._p2_to_p3_len = _.new_empty([patches_n, 1, primitives_n], requires_grad=True)
        self._b = _.new_empty([patches_n, spatial_dims_n, primitives_n], requires_grad=True)
        self._theta1 = _.new_empty([patches_n, 1, primitives_n], requires_grad=True)
        self._theta2 = _.new_empty([patches_n, 1, primitives_n], requires_grad=True)
        self._p2_to_p1_len.requires_grad_()
        self._p2_to_p3_len.requires_grad_()
        self._b.requires_grad_()
        self._theta1.requires_grad_()
        self._theta2.requires_grad_()
        self.canonical_parameters = [{'parameter': self._width, 'lr_factor': 1e-1},
                                     {'parameter': self._p2_to_p1_len, 'lr_factor': 1e-1},
                                     {'parameter': self._p2_to_p3_len, 'lr_factor': 1e-1},
                                     {'parameter': self._b, 'lr_factor': 1},
                                     {'parameter': self._theta1, 'lr_factor': 1e-1},
                                     {'parameter': self._theta2, 'lr_factor': 1e-1}]

        # auxiliary parameters
        self.p2_to_p1 = _.new_empty([patches_n, spatial_dims_n, primitives_n], requires_grad=False)
        self.p2_to_p3 = _.new_empty([patches_n, spatial_dims_n, primitives_n], requires_grad=False)
        self.tb = _.new_empty([patches_n, 1, primitives_n], requires_grad=False)
        self.b_to_p1_len = _.new_empty([patches_n, 1, primitives_n], requires_grad=False)
        self.b_to_p3_len = _.new_empty([patches_n, 1, primitives_n], requires_grad=False)
        self.old_p2_to_p1_len = _.new_empty([patches_n, 1, primitives_n], requires_grad=False)
        self.old_p2_to_p3_len = _.new_empty([patches_n, 1, primitives_n], requires_grad=False)
        self.p2_aux = None

        self.set_parameters(p1, p2, p3)

    # from .canonicals_with_probes import calculate_canonical_coordinates_with_probes, probe, probe_ts
    from .canonicals_with_cardano import calculate_canonical_coordinates_with_cardano
    # from .energy_with_quadratures import unit_energy_gauss5 as unit_energy
    from .energy_with_polyline import unit_energy

    @property
    def b(self):
        if self.pos_fixed:
            return self._b.data
        else:
            return self._b

    @property
    def b_to_p1(self):
        if self._b_to_p1 is None:
            self._b_to_p1 = torch.cat([torch.cos(self.theta1), torch.sin(self.theta1)], dim=1)
        return self._b_to_p1

    @property
    def b_to_p3(self):
        if self._b_to_p3 is None:
            self._b_to_p3 = torch.cat([torch.cos(self.theta2), torch.sin(self.theta2)], dim=1)
        return self._b_to_p3

    @property
    def p1(self):
        if self._p1 is None:
            if self.size_fixed:
                self._p1 = self.b + self.b_to_p1 * self.b_to_p1_len
            if self.pos_fixed:
                self._p1 = self.p2 + self.p2_to_p1 * self.p2_to_p1_len
        return self._p1

    @property
    def p3(self):
        if self._p3 is None:
            if self.size_fixed:
                self._p3 = self.b + self.b_to_p3 * self.b_to_p3_len
            if self.pos_fixed:
                self._p3 = self.p2 + self.p2_to_p3 * self.p2_to_p3_len
        return self._p3

    @property
    def p2(self):
        if self._p2 is None:
            if self.size_fixed:
                self._p2 = ((self.b - self.p3 * self.tb.pow(2) - self.p1 * (1 - self.tb).pow(2)) /
                            (self.tb * (1 - self.tb) * 2))
            if self.pos_fixed:
                return self.p2_aux
        return self._p2

    @property
    def p2_to_p1_len(self):
        if self.size_fixed:
            return self._p2_to_p1_len.data
        else:
            return self._p2_to_p1_len

    @property
    def p2_to_p3_len(self):
        if self.size_fixed:
            return self._p2_to_p3_len.data
        else:
            return self._p2_to_p3_len

    @property
    def theta1(self):
        if self.pos_fixed:
            return self._theta1.data
        else:
            return self._theta1

    @property
    def theta2(self):
        if self.pos_fixed:
            return self._theta2.data
        else:
            return self._theta2

    @property
    def width(self):
        if self.size_fixed:
            return self._width.data
        else:
            return self._width

    def calculate_canonical_coordinates(self, pixel_coords, from_midpoint=False, tol=1e-3, division_epsilon=1e-4):
        r"""

        Parameters
        ----------
        pixel_coords : torch.Tensor
            of shape [spatial_dims_n, pixels_n]

        Returns
        -------
        distance_to_curve : torch.Tensor
            of shape [patches_n, primitives_n, pixels_n]
        t_of_closest : torch.Tensor
            of shape [patches_n, primitives_n, pixels_n]
        """
        distance_to_curve, t_of_closest = self.calculate_canonical_coordinates_with_cardano(pixel_coords, tol=tol, division_epsilon=division_epsilon)
        if from_midpoint:
            patches_n, primitives_n = t_of_closest.shape[:2]
            t_of_closest = t_of_closest - self.tb.reshape(patches_n, primitives_n, 1)
        return distance_to_curve, t_of_closest

    def constrain_parameters(self, patch_width, patch_height, division_epsilon=division_epsilon):
        # 1. Constrain width and lengths to be non less than nonzero `min_linear`
        #    to prevent 'dying' of the lines (any position of a zero-sized line is optimal)
        self._width.data.clamp_(min=min_linear_size)
        self._p2_to_p1_len.data.clamp_(min=min_linear_size)
        self._p2_to_p3_len.data.clamp_(min=min_linear_size)

        # 2. Keep theta1 away from theta2 to prevent sharp folds
        t1 = self._theta1.data
        t2 = self._theta2.data
        dt = torch.remainder(t2 - t1 + np.pi, np.pi * 2) - np.pi  # in [-pi, pi)
        t_bisector = t1 + dt / 2
        del t1

        bisector_to_t1 = - dt / 2  # torch.remainder(t1 - t_bisector + np.pi, np.pi * 2) - np.pi
        del dt
        bisector_to_t1 = torch.where(bisector_to_t1 >= 0,
                                     bisector_to_t1.clamp(min=qbezier_min_fold_halfangle_radians),
                                     bisector_to_t1.clamp(max=qbezier_min_fold_halfangle_radians))
        torch.add(t_bisector, bisector_to_t1, out=self._theta1.data)
        del bisector_to_t1

        bisector_to_t2 = torch.remainder(t2 - t_bisector + np.pi, np.pi * 2) - np.pi
        del t2
        bisector_to_t2 = torch.where(bisector_to_t2 >= 0,
                                     bisector_to_t2.clamp(min=qbezier_min_fold_halfangle_radians),
                                     bisector_to_t2.clamp(max=qbezier_min_fold_halfangle_radians))
        torch.add(t_bisector, bisector_to_t2, out=self._theta2.data)
        del bisector_to_t2, t_bisector

        # 3. Swap width and length for short and wide 'dwarf' lines
        #    Blindly assume that P1->P2->P3 form straight line for these
        w = self._width.data
        l1 = self._p2_to_p1_len.data
        l2 = self._p2_to_p3_len.data
        l = l1 + l2
        dwarf_lines = w > l * dwarfness_ratio
        if dwarf_lines.any():
            w[dwarf_lines], l[dwarf_lines] = l[dwarf_lines], w[dwarf_lines]
            l1[dwarf_lines] = l[dwarf_lines] / 2
            l2[dwarf_lines] = l[dwarf_lines] / 2
            self._theta1.data[dwarf_lines] += np.pi / 2
            self._theta2.data[dwarf_lines] += np.pi / 2

            # 3.1. Swap moments etc for dwarf lines
            optimizer = self.optimizer
            w = self._width
            l1 = self._p2_to_p1_len
            l2 = self._p2_to_p3_len
            if isinstance(optimizer, torch.optim.Adam) and 'exp_avg' in optimizer.state[w]:
                w_avg = optimizer.state[w]['exp_avg'].data
                w_sq = optimizer.state[w]['exp_avg_sq'].data
                l1_avg = optimizer.state[l1]['exp_avg'].data
                l1_sq = optimizer.state[l1]['exp_avg_sq'].data
                l2_avg = optimizer.state[l2]['exp_avg'].data
                l2_sq = optimizer.state[l2]['exp_avg_sq'].data

                w_avg_halved = w_avg[dwarf_lines] / 2
                w_avg[dwarf_lines], l1_avg[dwarf_lines], l2_avg[dwarf_lines] = \
                    l1_avg[dwarf_lines] + l2_avg[dwarf_lines], w_avg_halved, w_avg_halved
                del w_avg_halved

                w_sq_halved = w_sq[dwarf_lines] / 2
                w_sq[dwarf_lines], l1_sq[dwarf_lines], l2_sq[dwarf_lines] = \
                    l1_sq[dwarf_lines] + l2_sq[dwarf_lines], w_sq_halved, w_sq_halved
                del w_sq_halved

        # 4. Keep thetas in [0, 2pi)
        self._theta1.data.remainder_(np.pi * 2)
        self._theta2.data.remainder_(np.pi * 2)

        # 5. Limit positions of the curves in the canvas
        #    Nonzero padding is used to prevent nonstability for the curves trying to fit super-narrow raster
        #    (i.e, with small shading value) at the very edge of the canvas
        self._b.data[:, 0].clamp_(min=-coordinates_constrain_padding,
                                  max=patch_width + coordinates_constrain_padding)
        self._b.data[:, 1].clamp_(min=-coordinates_constrain_padding,
                                  max=patch_height + coordinates_constrain_padding)

        # 6. Limit the lengths w.r.t the edges of the canvas
        # FIXME: some bug here
        # def do_this(b, l, t):
        #     cos = t[:, 0].cos()
        #     max_x_lim = ((patch_width + coordinates_constrain_padding - b[:, 0]) /
        #                  (cos + division_epsilon).where(cos >= 0, cos.new_full([], np.inf)))
        #     min_x_lim = ((coordinates_constrain_padding + b[:, 0]) /
        #                  (cos.abs() + division_epsilon).where(cos < 0, cos.new_full([], np.inf)))
        #     del cos
        #
        #     sin = t[:, 0].sin()
        #     max_y_lim = ((patch_height + coordinates_constrain_padding - b[:, 1]) /
        #                  (sin + division_epsilon).where(sin >= 0, sin.new_full([], np.inf)))
        #     min_y_lim = ((coordinates_constrain_padding + b[:, 1]) /
        #                  (sin.abs() + division_epsilon).where(sin < 0, sin.new_full([], np.inf)))
        #     del sin
        #
        #     max_l = torch.min(torch.stack([max_x_lim, min_x_lim, max_y_lim, min_y_lim], dim=0), dim=0)[0]
        #     constrained_l = torch.min(l, max_l.unsqueeze(1)).clamp(min=min_linear_size)
        #     l.copy_(constrained_l)
        # do_this(self._b.data, self.b_to_p1_len, self._theta1.data)
        # do_this(self._b.data, self.b_to_p3_len, self._theta2.data)

    def get_derivatives(self, t):
        r"""

        Parameters
        ----------
        t : torch.Tensor
            of shape [t_n] and corresponding type and device

        Returns
        -------
        points : torch.Tensor
            of shape [t_n, patches_n, spatial_dims_n, primitives_n]
        """
        t = t.reshape(-1, 1, 1, 1)
        return ((self.p2 - self.p1) * (1 - t) + (self.p3 - self.p2) * t) * 2

    def get_neighbourhood_weighting(self, pixel_coords, pixel_values, empty_pixel_tol=empty_pixel_tolerance,
                                    elementary_halfwidth=elementary_halfwidth, x_padding=neighbourhood_padding,
                                    y_padding=qbezier_y_neighbourhood_padding):
        r"""

        Parameters
        ----------
        pixel_coords : torch.Tensor
            of shape [spatial_dims_n, pixels_n]
        pixel_values : torch.Tensor
            of shape [patches_n, pixels_n]

        Returns
        -------
        in_neighbourhood : torch.Tensor
            Tensor of shape [patches_n, primitives_n, pixels_n] with True for pixels in neighbourhood of the primitive
        """
        return super().get_neighbourhood_weighting(
            pixel_coords, pixel_values, empty_pixel_tol=empty_pixel_tol, elementary_halfwidth=elementary_halfwidth,
            x_padding=x_padding, y_padding=y_padding)

    def get_points(self, t):
        r"""

        Parameters
        ----------
        t : torch.Tensor
            of shape [t_n, patches_n, primitives_n] or broadcastable

        Returns
        -------
        points : torch.Tensor
            of shape [t_n, patches_n, spatial_dims_n, primitives_n]
        """
        spatial_dim_i = 2
        t = t.unsqueeze(spatial_dim_i).type(self.dtype).to(self.device)
        return qbezier_point(self.p1, self.p2, self.p3, t)

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

        p1 = self.p1.data
        spatial_dims_n = p1.shape[1]
        p1 = p1.reshape(patches_n, spatial_dims_n, primitives_n, 1)
        p2 = self.p2.data.reshape(patches_n, spatial_dims_n, primitives_n, 1)
        p3 = self.p3.data.reshape(patches_n, spatial_dims_n, primitives_n, 1)

        t = t.reshape(patches_n, 1, primitives_n, pixels_n)

        vector_field = (p2 - p1) * (1 - t)
        del p1
        vector_field += (p3 - p2) * t
        del t, p2, p3
        vector_field = torch.nn.functional.normalize(vector_field, dim=1)
        return vector_field

    def invalidate_pos_dependent(self):
        self._p1 = None
        self._p2 = None
        self._p3 = None
        self._b_to_p1 = None
        self._b_to_p3 = None
        self._unit_energy = None
        self._q_collinearity = None

    def invalidate_size_dependent(self):
        self._p1 = None
        self._p2 = None
        self._p3 = None
        self._b_to_p1 = None
        self._b_to_p3 = None
        self._unit_energy = None
        self._q_collinearity = None

    def merge_close(self):
        p1 = self.p1.data.cpu().clone()
        p2 = self.p2.data.cpu().clone()
        p3 = self.p3.data.cpu().clone()
        w = self._width.data.cpu().clone()

        device = self.device
        patches_n = self.patches_n
        primitives_n = self.primitives_n

        patches = torch.cat([p1, p2, p3, w], 1).permute(0, 2, 1)
        joined_curves = p1.new_zeros([patches_n, primitives_n, 7])
        joined_curves[:, :, 2] = reinit_initial_length / 2
        joined_curves[:, :, 4] = reinit_initial_length
        joined_curves[:, :, 6] = reinit_initial_width

        for i, patch in enumerate(patches):
            common_width_in_patch = np.percentile(patch[:, -1], 90)
            join_tol = .5 * common_width_in_patch
            fit_tol = .5 * common_width_in_patch
            w_tol = np.inf

            _ = join_quad_beziers(patch, join_tol=join_tol, fit_tol=fit_tol, w_tol=w_tol)
            _primitives_n = len(_)
            joined_curves[i, :_primitives_n] = _
        joined_curves = joined_curves.permute(0, 2, 1).contiguous()
        p1 = joined_curves[:, :2]
        p2 = joined_curves[:, 2:4]
        p3 = joined_curves[:, 4:6]
        w = joined_curves[:, 6:7]

        self._width.data.copy_(w.to(device))
        self.set_parameters(p1, p2, p3)

        optimizer = self.optimizer
        if isinstance(optimizer, torch.optim.Adam) and 'exp_avg' in optimizer.state[self._width]:
            for p in self.canonical_parameters:
                p = p['parameter']
                if (p in optimizer.state) and ('exp_avg' in optimizer.state[p]):
                    optimizer.state[p]['exp_avg'].zero_()
                    optimizer.state[p]['exp_avg_sq'].zero_()
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            # optimizer.state[self._theta1]['exp_avg_sq'].data.fill_(1)
            # optimizer.state[self._theta2]['exp_avg_sq'].data.fill_(1)


    def reinit_primitives(self, primitives_to_reinit, coords_to_reinit_at, initial_width, initial_length):
        r"""

        Parameters
        ----------
        primitives_to_reinit : torch.Tensor
            of shape [patches_n, primitives_n]
        coords_to_reinit_at : torch.Tensor or None
            of shape [*, spatial_dims_n]
        """
        p1 = self.p1.data
        p2 = self.p2.data
        p3 = self.p3.data

        # 1. Put p2 at coords_to_reinit
        patches_n, spatial_dims_n, primitives_n = p2.shape
        primitives_to_reinit = primitives_to_reinit.reshape(patches_n, 1, primitives_n)
        if coords_to_reinit_at is not None:
            p2.masked_scatter_(primitives_to_reinit, coords_to_reinit_at)
        else:
            # keep in place
            coords_to_reinit_at = p2[primitives_to_reinit.expand_as(p2)].reshape(-1, spatial_dims_n)

        # 2. Put p1 and p3 at `initial_length / 2` distances from p2
        dp = coords_to_reinit_at.new_tensor([[initial_length / 2, 0]])
        p1_coords = coords_to_reinit_at - dp
        p1.masked_scatter_(primitives_to_reinit, p1_coords)
        del p1_coords

        p3_coords = coords_to_reinit_at + dp
        p3.masked_scatter_(primitives_to_reinit, p3_coords)
        del p3_coords, dp

        # 3. Reinit widths
        self._width.data.masked_fill_(primitives_to_reinit, initial_width)

        # 4. Update parameters
        self.set_parameters(p1, p2, p3)

        # 5. Reset optimizer moments
        optimizer = self.optimizer
        if isinstance(optimizer, torch.optim.Adam):
            for p in self.canonical_parameters:
                p = p['parameter']
                if 'exp_avg' in optimizer.state[p]:
                    optimizer.state[p]['exp_avg'].data.masked_fill_(primitives_to_reinit, 0)
                    optimizer.state[p]['exp_avg_sq'].data.masked_fill_(primitives_to_reinit, 0)

    def render_single_primitive_with_cairo(self, ctx, patch_i, primitive_i, min_width=0):
        width = self.width[patch_i, 0, primitive_i].detach().cpu().numpy()
        if width < min_width:
            return

        q1 = self.p1[patch_i, :, primitive_i].detach().cpu().numpy()
        q2 = self.p2[patch_i, :, primitive_i].detach().cpu().numpy()
        q3 = self.p3[patch_i, :, primitive_i].detach().cpu().numpy()

        # convert to cubic
        c1 = q1
        c4 = q3
        c2 = (q2 * 2 + q1) / 3
        c3 = (q2 * 2 + q3) / 3

        # draw
        ctx.move_to(*c1)
        ctx.curve_to(*c2, *c3, *c4)
        ctx.set_line_width(width)
        ctx.stroke()

    def render_skeleton_total(self, width, height, visibility_width_threshold=visibility_width_threshold, scaling=4,
                              line_color=(31 / 255, 119 / 255, 180 / 255), line_width=2, control_line_width=1,
                              node_color=(255 / 255, 127 / 255, 14 / 255), node_size=4, control_node_size=2,
                              controls_color=(214 / 255, 39 / 255, 40 / 255)):
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

        pars = (np.concatenate([self.p1.detach().cpu().numpy(), self.p2.detach().cpu().numpy(),
                               self.p3.detach().cpu().numpy(), widths, self._b.detach().cpu().numpy()], 1)
                  .swapaxes(1, 2))
        pars *= scaling

        return np.stack([render_with_skeleton(
            {PT_QBEZIER_B: patch[visibility]}, [width * scaling, height * scaling], data_representation='vahe',
            linecaps=refinement_linecaps, linejoin=refinement_linejoin, line_color=line_color, line_width=line_width,
            node_color=node_color, node_size=node_size,
            control_line_width=control_line_width, control_node_size=control_node_size, controls_color=controls_color)
            for patch, visibility in zip(pars, visible)])

    def set_parameters(self, p1, p2, p3):
        p1 = p1.data.type(self.dtype).to(self.device)
        p2 = p2.data.type(self.dtype).to(self.device)
        p3 = p3.data.type(self.dtype).to(self.device)

        self.p2_to_p1 = p1 - p2
        torch.norm(self.p2_to_p1, dim=1, keepdim=True, out=self._p2_to_p1_len.data)
        self.p2_to_p1 /= (self._p2_to_p1_len.data + division_epsilon)
        self.old_p2_to_p1_len.copy_(self._p2_to_p1_len.data)

        self.p2_to_p3 = p3 - p2
        torch.norm(self.p2_to_p3, dim=1, keepdim=True, out=self._p2_to_p3_len.data)
        self.p2_to_p3 /= (self._p2_to_p3_len.data + division_epsilon)
        self.old_p2_to_p3_len.copy_(self._p2_to_p3_len.data)

        _ = torch.sqrt(self._p2_to_p1_len.data)
        self.tb = _ / (_ + torch.sqrt(self._p2_to_p3_len.data))
        del _

        tb = self.tb
        torch.add(p1 * (1 - tb).pow(2) + p2 * (1 - tb) * tb * 2, p3 * tb.pow(2), out=self._b.data)

        b_to_p1 = p1 - self._b.data
        self.b_to_p1_len = torch.norm(b_to_p1, dim=1, keepdim=True)
        torch.atan2(b_to_p1[:, 1:2], b_to_p1[:, :1], out=self._theta1.data)

        b_to_p3 = p3 - self._b.data
        self.b_to_p3_len = torch.norm(b_to_p3, dim=1, keepdim=True)
        torch.atan2(b_to_p3[:, 1:2], b_to_p3[:, :1], out=self._theta2.data)

        self.p2_aux = p2

        self._segments_p1 = None
        self._segments_dir = None
        self._segments_length = None

        self.invalidate_pos_dependent()
        self.invalidate_size_dependent()

    def synchronize_parameters(self, division_epsilon=division_epsilon, max_dt_end=qbezier_max_dt_end):
        r"""Synchronize the auxiliary parameters of the curve tensor with the main, optimized parameters.
        Call each time the main parameters are updated.

        Notes
        -----
        The algorithm is laced with assumption that everything changes only a little.
        It simulates separate updates, first, to position parameters and then to size parameters.
        """
        super().synchronize_parameters()

        # 1. Simulate update of position parameters with fixed size.
        #    Calculate P1, P2 and P3 from new B, theta1, and theta2 and old lengths of B->P1, B->P2.
        spatial_dim_i = 1
        b = self._b.data
        t1 = self._theta1.data
        b_to_p1 = torch.cat([torch.cos(t1), torch.sin(t1)], dim=spatial_dim_i)
        p1 = b + b_to_p1 * self.b_to_p1_len
        del t1, b_to_p1

        t2 = self._theta2.data
        b_to_p3 = torch.cat([torch.cos(t2), torch.sin(t2)], dim=spatial_dim_i)
        p3 = b + b_to_p3 * self.b_to_p3_len
        del t2, b_to_p3

        tb = self.tb
        p2 = ((b - p3 * tb.pow(2) - p1 * (1 - tb).pow(2)) / (tb * (1 - tb) * 2))
        del b, tb

        # 2. Simulate update of size parameters, i.e lengths.
        #
        # calculate left dt
        dl1 = self._p2_to_p1_len.data - self.old_p2_to_p1_len
        dl1_over_dt = torch.norm(p2 - p1, dim=spatial_dim_i, keepdim=True) * 2
        dt1 = -dl1 / (dl1_over_dt + division_epsilon)
        del dl1, dl1_over_dt
        dt1.clamp_(min=-max_dt_end, max=max_dt_end)

        # calculate right dt
        dl2 = self._p2_to_p3_len.data - self.old_p2_to_p3_len
        dl2_over_dt = torch.norm(p3 - p2, dim=spatial_dim_i, keepdim=True) * 2
        dt2 = dl2 / (dl2_over_dt + division_epsilon)
        del dl2, dl2_over_dt
        dt2.clamp_(min=-max_dt_end, max=max_dt_end)

        # update points
        new_p1 = qbezier_point(p1, p2, p3, dt1)
        new_p3 = qbezier_point(p1, p2, p3, 1 + dt2)
        new_p2 = (qbezier_point(p1, p2, p3, (dt1 + 1 + dt2) / 2) - (new_p1 + new_p3) / 4) * 2
        del p1, p2, p3, dt1, dt2

        # 3. Update the main and auxiliary parameters w.r.t these P1, P2, P3
        self.set_parameters(new_p1, new_p2, new_p3)


def qbezier_point(p1, p2, p3, t):
    omt = 1 - t
    return p1 * omt.pow(2) + p2 * omt * t * 2 + p3 * t.pow(2)
