import torch

from ...parameters import division_epsilon
from ..line_tensor.procedures import calculate_canonical_coordinates


def unit_energy(self, pixel_coords, division_epsilon=division_epsilon):
    r"""

    Parameters
    ----------
    pixel_coords : torch.Tensor
        of shape [spatial_dims_n, pixels_n]
    division_epsilon

    Returns
    -------
    energies : torch.Tensor
        of shape [patches_n, primitives_n, pixels_n]
    """
    spatial_dims_n, pixels_n = pixel_coords.shape
    patches_n = self.patches_n
    primitives_n = self.primitives_n

    if self._segments_p1 is None:
        # 1. Find good split for each curve
        # TODO maybe adaptively
        t = torch.linspace(0, 1, steps=10, dtype=self.dtype, device=self.device).reshape(-1, 1, 1)

        # 2. Split each curve into line segments
        segments_n = len(t) - 1
        spatial_dim_i = 2
        endpoints = self.get_points(t)  # [segments_n + 1, patches_n, spatial_dims_n, primitives_n]
        del t
        p1 = endpoints[:segments_n]
        dir = endpoints[1:] - p1
        del endpoints
        length = dir.norm(dim=spatial_dim_i, keepdim=True)
        dir = dir / (length + division_epsilon)
        length = length.data  # we don't want to optimize length here

        self._segments_p1 = p1
        self._segments_dir = dir
        self._segments_length = length
    else:
        p1 = self._segments_p1
        dir = self._segments_dir
        length = self._segments_length
        segments_n = len(p1)

    if self.pos_fixed:
        p1 = p1.data
        dir = dir.data

    # 3. Calculate segment-to-pixel energies
    p1 = p1.reshape(-1, spatial_dims_n, primitives_n)
    dir = dir.reshape(-1, spatial_dims_n, primitives_n)
    canonical_x, canonical_y = calculate_canonical_coordinates(pixel_coords, p1, dir)
    del p1, dir
    canonical_x = canonical_x.reshape(segments_n, patches_n, primitives_n, pixels_n)
    canonical_y = canonical_y.reshape(segments_n, patches_n, primitives_n, pixels_n)

    halfwidth = self.width.reshape(1, patches_n, primitives_n, 1) / 2
    length = length.reshape(segments_n, patches_n, primitives_n, 1)
    energies = self.energy_procedures.unit_energy_line_to_canonical_point(halfwidth, length, canonical_x, canonical_y)
    del length, canonical_x, canonical_y

    # 4. Sum over segments
    segments_dim_i = 0
    energies = energies.sum(segments_dim_i)

    # 5. Inject dependency on p2-p1 and p2-p3 lengths using a trick
    if not self.size_fixed:
        segments_n = 2
        p1 = torch.stack([self.p1.data, self.p3.data])
        dir = torch.stack([self.p2_to_p1, self.p2_to_p3])
        length = torch.stack([self.p2_to_p1_len - self.p2_to_p1_len.data, self.p2_to_p3_len - self.p2_to_p3_len.data])

        p1 = p1.reshape(-1, spatial_dims_n, primitives_n)
        dir = dir.reshape(-1, spatial_dims_n, primitives_n)
        canonical_x, canonical_y = calculate_canonical_coordinates(pixel_coords, p1, dir)
        del p1, dir
        canonical_x = canonical_x.reshape(segments_n, patches_n, primitives_n, pixels_n)
        canonical_y = canonical_y.reshape(segments_n, patches_n, primitives_n, pixels_n)

        length = length.reshape(segments_n, patches_n, primitives_n, 1)
        halfwidth = halfwidth.data
        len_dependent_terms = self.energy_procedures.unit_energy_line_to_canonical_point(
            halfwidth, length, canonical_x, canonical_y)
        del canonical_x, canonical_y, length

        segments_dim_i = 0
        energies = energies + len_dependent_terms.sum(segments_dim_i)
    del halfwidth

    # # 6. Add curvature penalty
    # if not self.pos_fixed:
    #     # # dtheta in [-pi, pi]
    #     # dtheta = torch.remainder(self.theta2 - self.theta1 + np.pi, np.pi * 2) - np.pi
    #     #
    #     # beta = 1 / (15 * np.pi / 180) ** 2
    #     # amplitude = 1e6
    #     # curvature_penalty = torch.exp(-dtheta.pow(2) * beta) * amplitude
    #     # del dtheta, beta
    #
    #     # dtheta in [-pi, pi]
    #     dtheta = torch.remainder(self.theta2 - self.theta1.data + np.pi, np.pi * 2) - np.pi
    #
    #     beta = 1 / (15 * np.pi / 180) ** 2
    #     amplitude = 1e6
    #     curvature_penalty = torch.exp(-dtheta.pow(2) * beta) * amplitude
    #     del dtheta, beta
    #
    #     curvature_penalty.reshape(patches_n, primitives_n, 1)
    #     energies = energies + curvature_penalty
    #     del curvature_penalty

    # # draw for debugging
    # import numpy as np
    # _ = energies
    # _ = _[:, 0].detach().cpu()
    #
    # h = int(np.sqrt(pixels_n))
    # assert pixels_n % h == 0
    # w = pixels_n // h
    # _ = _.reshape(patches_n, h, w)
    # self.debug_ax.imshow(np.vstack(_))

    return energies
