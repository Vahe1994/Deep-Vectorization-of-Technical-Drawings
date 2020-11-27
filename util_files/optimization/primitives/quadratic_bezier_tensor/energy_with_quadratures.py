import torch

from ...parameters import division_epsilon


def unit_energy_with_quadrature(self, pixel_coords, stencil_t, stencil_s,
                                weights, division_epsilon=division_epsilon):
    r"""

    Parameters
    ----------
    self
    pixel_coords : torch.Tensor
        of shape [spatial_dims_n, pixels_n] and corresponding type and device
    stencil_t : torch.Tensor
        of shape [t_n] and corresponding type and device
    stencil_s : torch.Tensor
        of shape [s_n] and corresponding type and device
    weights : torch.Tensor
        of shape [s_n, t_n] and corresponding type and device.
        Weights include Jacobian.
    division_epsilon

    Returns
    -------
    energies : torch.Tensor
        of shape [patches_n, primitives_n, pixels_n]
    """
    spatial_dims_n, pixels_n = pixel_coords.shape
    patches_n = self.patches_n
    primitives_n = self.primitives_n
    s_n, t_n = weights.shape
    width = self.width

    # 1. calculate stencil points
    spatial_dim_i = 2
    oncurves = self.get_points(stencil_t.reshape(-1, 1, 1))  # [t_n, patches_n, spatial_dims_n, primitives_n]
    derivs = self.get_derivatives(stencil_t)
    del stencil_t
    deriv_norms = derivs.norm(dim=spatial_dim_i, keepdim=True)
    normals = torch.stack([-derivs[:, :, 1], derivs[:, :, 0]], dim=spatial_dim_i) / (deriv_norms + division_epsilon)
    del derivs
    width = width.reshape(1, patches_n, 1, primitives_n)
    normals = normals * width  # not quite normals already
    normals = normals.reshape(1, t_n, patches_n, spatial_dims_n, primitives_n)
    stencil_s = stencil_s.reshape(s_n, 1, 1, 1, 1)
    normal_shifts = normals * stencil_s
    del stencil_s, normals
    oncurves = oncurves.reshape(1, t_n, patches_n, spatial_dims_n, primitives_n)
    stencil_points = oncurves + normal_shifts  # [s_n, t_n, patches_n, spatial_dims_n, primitives_n]
    del oncurves, normal_shifts

    # 2. shift stencil points w.r.t each pixel
    stencil_points = stencil_points.reshape(s_n, t_n, patches_n, spatial_dims_n, primitives_n, 1)
    pixel_coords = pixel_coords.reshape(1, 1, 1, spatial_dims_n, 1, pixels_n)
    stencil_points = stencil_points - pixel_coords
    del pixel_coords

    # 3. calculate values_of_the_potential
    spatial_dim_i = 3
    r2 = stencil_points.pow(2).sum(spatial_dim_i)
    del stencil_points
    energies = self.energy_procedures.unit_energy_point_to_point(r2)

    # 4. add weighting and sum
    s_dim_i = 0
    weights = weights.reshape(s_n, t_n, 1, 1, 1)
    energies = energies * weights
    energies = energies.sum(s_dim_i)

    t_dim_i = 0
    deriv_norms = deriv_norms.reshape(t_n, patches_n, primitives_n, 1)
    energies = energies * deriv_norms
    del deriv_norms
    energies = energies.sum(t_dim_i)
    width = width.reshape(patches_n, primitives_n, 1)
    energies = energies * width
    del width

    # draw for debugging
    import numpy as np
    _ = energies
    _ = _[:, 0].detach().cpu()

    h = int(np.sqrt(pixels_n))
    assert pixels_n % h == 0
    w = pixels_n // h
    _ = _.reshape(patches_n, h, w)
    self.debug_ax.imshow(np.vstack(_))

    return energies


def unit_energy_gauss5(self, pixel_coords, division_epsilon=division_epsilon):
    r"""

    Parameters
    ----------
    self
    pixel_coords : torch.Tensor
        of shape [spatial_dims_n, pixels_n] and corresponding type and device
    division_epsilon

    Returns
    -------
    energies : torch.Tensor
        of shape [patches_n, primitives_n, pixels_n]
    """
    _ = torch.tensor([], dtype=self.dtype, device=self.device)

    _1 = 2 * (10. / 7) ** .5
    _2 = ((5 - _1) ** .5) / 3
    _3 = ((5 + _1) ** .5) / 3
    stencil_t = _.new_tensor([0, -_2, _2, -_3, _3])
    stencil_t = (stencil_t + 1) / 2

    stencil_s = _.new_tensor([-1 / 2, 0, 1 / 2])  # simpson w.r.t s
    jacobi_det = 1/4
    weights_s = _.new_tensor([1, 4, 1]).reshape(len(stencil_s), 1) / 3

    _1 = 13 * 70 ** .5
    _2 = 322 + _1
    _3 = 322 - _1
    weights_t = _.new_tensor([512, _2, _2, _3, _3]).reshape(1, len(stencil_t)) / 900

    weights = weights_s * weights_t * jacobi_det
    del _

    return unit_energy_with_quadrature(self, pixel_coords, stencil_t, stencil_s,
                                       weights, division_epsilon=division_epsilon)


def unit_energy_simpson(self, pixel_coords, division_epsilon=division_epsilon):
    r"""

    Parameters
    ----------
    self
    pixel_coords : torch.Tensor
        of shape [spatial_dims_n, pixels_n] and corresponding type and device
    division_epsilon

    Returns
    -------
    energies : torch.Tensor
        of shape [patches_n, primitives_n, pixels_n]
    """
    _ = torch.tensor([], dtype=self.dtype, device=self.device)
    stencil_t = _.new_tensor([0, .5, 1])
    stencil_s = _.new_tensor([-1 / 2, 0, 1 / 2])
    jacobi_det = 1/4
    weights = _.new_tensor([1, 4, 1]).reshape(3, 1) * _.new_tensor([1, 4, 1]).reshape(1, 3) / 9 * jacobi_det
    del _

    return unit_energy_with_quadrature(self, pixel_coords, stencil_t, stencil_s,
                                       weights, division_epsilon=division_epsilon)
