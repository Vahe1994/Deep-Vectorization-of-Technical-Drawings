import numpy as np
import torch


def calculate_canonical_coordinates_with_cardano(self, pixel_coords, tol=1e-3, division_epsilon=1e-4):
    r"""

    Parameters
    ----------
    pixel_coords : torch.Tensor
        of shape [spatial_dims_n, pixels_n]
    tol : scalar
        If imaginary part of a complex number is less than `tol` then the number is considered real.
        Used for root selection.

    Returns
    -------
    distance_to_curve : torch.Tensor
        of shape [patches_n, primitives_n, pixels_n]
    t_of_closest : torch.Tensor
        of shape [patches_n, primitives_n, pixels_n]
    """
    p1 = self.p1.data
    p2 = self.p2.data
    p3 = self.p3.data
    pixels_n = pixel_coords.shape[1]
    patches_n, spatial_dims_n, primitives_n = p1.shape

    p1 = p1.reshape(patches_n, spatial_dims_n, primitives_n, 1)
    p2 = p2.reshape(patches_n, spatial_dims_n, primitives_n, 1)
    p3 = p3.reshape(patches_n, spatial_dims_n, primitives_n, 1)
    pixel_coords = pixel_coords.reshape(1, spatial_dims_n, 1, pixels_n)
    p1, p2, p3, pixel_coords = torch.broadcast_tensors(p1, p2, p3, pixel_coords)

    p1 = p1 - pixel_coords
    p2 = p2 - pixel_coords
    p3 = p3 - pixel_coords
    del pixel_coords

    A = p2 - p1
    B = p3 - p2 - A
    B = B.where(B.abs() >= division_epsilon, B.new_full([], division_epsilon))

    spatial_dim_i = 1
    a = B.pow(2).sum(dim=spatial_dim_i)
    b = (A * B).sum(dim=spatial_dim_i) * 3
    c = A.pow(2).sum(dim=spatial_dim_i) * 2 + (p1 * B).sum(dim=spatial_dim_i)
    d = (p1 * A).sum(dim=spatial_dim_i)
    del A, B

    delta = - b / (a * 3)
    c_over_a = c / a

    p = c_over_a - delta.pow(2) * 3
    q = d / a - delta.pow(3) * 2 + delta * c_over_a
    del a, b, c, d, c_over_a

    Q = (p / 3).pow(3) + (q / 2).pow(2)

    Q_abs_sqrt = Q.abs().sqrt()
    Q_nonneg = Q >= 0
    del Q

    halved_q = q / 2
    del q
    flip_sign = Q_nonneg & (Q_abs_sqrt == halved_q)
    Q_abs_sqrt = (-Q_abs_sqrt).where(flip_sign, Q_abs_sqrt)
    del flip_sign

    alpha_cube_re = Q_abs_sqrt.where(Q_nonneg, Q_abs_sqrt.new_zeros([])) - halved_q
    alpha_cube_im = Q_abs_sqrt.where(~Q_nonneg, Q_abs_sqrt.new_zeros([]))
    del Q_abs_sqrt, Q_nonneg, halved_q

    alpha_modulus = (alpha_cube_re.pow(2) + alpha_cube_im.pow(2)).pow(1/6)
    alpha_phase = torch.atan2(alpha_cube_im, alpha_cube_re) / 3
    del alpha_cube_re, alpha_cube_im

    sols = alpha_modulus.new_empty([3, patches_n, primitives_n, pixels_n])
    for i in range(3):
        alpha_phase_shifted = alpha_phase + np.pi * 2 * i / 3
        p_over_three_alpha_m = p / (alpha_modulus * 3)
        sol_im = (alpha_modulus + p_over_three_alpha_m) * alpha_phase_shifted.sin()
        real_sol = sol_im.abs() < tol
        sol_re = (alpha_modulus - p_over_three_alpha_m) * alpha_phase_shifted.cos()
        sols[i] = sol_re.where(real_sol, sol_re.new_full([], np.inf))
    del p, alpha_modulus, alpha_phase, alpha_phase_shifted, p_over_three_alpha_m, sol_im, real_sol, sol_re

    t_of_closest = sols + delta
    del delta, sols

    t = t_of_closest.reshape(3, patches_n, 1, primitives_n, pixels_n)
    omt = 1 - t
    p1 = p1.reshape(1, patches_n, spatial_dims_n, primitives_n, pixels_n)
    p2 = p2.reshape(1, patches_n, spatial_dims_n, primitives_n, pixels_n)
    p3 = p3.reshape(1, patches_n, spatial_dims_n, primitives_n, pixels_n)
    closest_on_curve = p1 * omt.pow(2) + p2 * omt * t * 2 + p3 * t.pow(2)
    del t, omt, p1, p2, p3

    spatial_dim_i = 2
    dist_to_curve = closest_on_curve.norm(dim=spatial_dim_i)
    dist_to_curve.masked_fill_(torch.isnan(dist_to_curve), np.inf)

    del closest_on_curve
    dist_to_curve, sol_id = dist_to_curve.min(dim=0)

    sol_id = sol_id.reshape(1, patches_n, primitives_n, pixels_n)
    t_of_closest = torch.gather(t_of_closest, dim=0, index=sol_id)[0]
    del sol_id

    # draw coordinates for debugging
    # h = int(np.sqrt(pixels_n))
    # assert pixels_n % h == 0
    # w = pixels_n // h
    # _ = t_of_closest.cpu()
    # _ = _.reshape(patches_n, h, w)
    # self.debug_ax.imshow(np.vstack(_))

    return dist_to_curve, t_of_closest
