import numpy as np
import torch


def log_debug(s):
    pass
    # print(s)


def join_quad_beziers(curves, join_tol=.5, fit_tol=.5, w_tol=.5):
    r"""

    Parameters
    ----------
    curves : array_like
        of shape [curves_n, control_points_n * spatial_dims_n + 1]

    Returns
    -------
    curves : torch.Tensor
        of shape [new_curves_n, control_points_n * spatial_dims_n + 1]
    """
    spatial_dims_n = 2

    previous_curves_n = len(curves)
    curves = {i: torch.as_tensor(curves[i]) for i in range(previous_curves_n)}

    source_points = dict()
    for i in range(previous_curves_n):
        curve = curves[i]
        p1 = curve[:spatial_dims_n]
        p2 = curve[spatial_dims_n:spatial_dims_n * 2]
        p3 = curve[spatial_dims_n * 2:spatial_dims_n * 3]
        _ = curve[spatial_dims_n * 3]
        _, pb = bisector_point(p1, p2, p3)
        source_points[i] = [p1.tolist(), pb.tolist(), p3.tolist()]

    maxi = previous_curves_n
    while True:
        # 1. For every this curve
        for i in range(maxi):
            if i not in curves:
                continue

            curve = curves[i]
            p1 = curve[:spatial_dims_n]
            p2 = curve[spatial_dims_n:spatial_dims_n * 2]
            p3 = curve[spatial_dims_n * 2:spatial_dims_n * 3]
            w = curve[spatial_dims_n * 3]
            tb, pb = bisector_point(p1, p2, p3)

            # 2. For every that curve
            for j in range(maxi):
                if (j not in curves) or (j == i):
                    continue
                that_curve = curves[j]
                that_w = that_curve[spatial_dims_n * 3]

                # 3. If widths are different -- don't fuse
                if (that_w - w).abs() > w_tol:
                    log_debug(f'{i}-{j}: Widths are different')
                    continue

                q1 = that_curve[:spatial_dims_n]
                q2 = that_curve[spatial_dims_n:spatial_dims_n * 2]
                q3 = that_curve[spatial_dims_n * 2:spatial_dims_n * 3]
                sb, qb = bisector_point(q1, q2, q3)
                del q2

                # 4. Calculate distances to this curve from q1, qb, q3 and their ts
                p1 = p1.reshape(1, spatial_dims_n, 1)
                p2 = p2.reshape(1, spatial_dims_n, 1)
                p3 = p3.reshape(1, spatial_dims_n, 1)
                coords_q = torch.stack([q1, qb, q3], dim=1)
                dq, tq = calculate_canonical_coordinates_with_cardano(p1, p2, p3, coords_q, real_distance=True)
                del coords_q
                p1 = p1.reshape(spatial_dims_n)
                p2 = p2.reshape(spatial_dims_n)
                p3 = p3.reshape(spatial_dims_n)
                dq1, dqb, dq3 = dq.reshape(3)
                tq1, tqb, tq3 = tq.reshape(3)
                del tq, tqb

                # 5. If all three are close -- fuse that curve into this curve
                if (dq <= join_tol).all():
                    log_debug(f'{i}-{j}: All three are close')
                    del curves[j]
                    del source_points[j]
                    continue

                # 6. If all far, or
                #       ends are close but bisector point is far, or vice versa -- skip
                if ((dq > join_tol).all() or
                        ((dq1 <= join_tol) and (dq3 <= join_tol) and (dqb > join_tol)) or
                        ((dq1 > join_tol) and (dq3 > join_tol) and (dqb <= join_tol))):
                    log_debug(f'{i}-{j}: Won\'t merge')
                    continue
                del dq

                #    Else, either q1 or q3 is far from this curve
                # 7. Sort the points of that curve so that the closest is q1
                if dq3 < dq1:
                    q1, q3 = q3, q1
                    tq1, tq3 = tq3, tq1
                    sb = 1 - sb
                del dq1, dq3

                # 8. Sort the points of this curve so that the points go from p1 to q3, i.e tq3 > 1
                if tq3 < 0:
                    p1, p3 = p3, p1
                    tb = 1 - tb
                    tq1 = 1 - tq1
                    tq3 = 1 - tq3

                # 8.5.
                eps = .05
                if (tq1 < eps) or (tq3 < 1 + eps):
                    log_debug(f'{i}-{j}: Won\'t merge')
                    continue
                del tq3

                # 9. Find best fit
                fit, error = find_qbezier_best_join_fit(p1, pb, p3, q1, qb, q3, tb, sb, tq1)
                del q1, qb, q3, sb, tq1
                if error > fit_tol:
                    log_debug(f'{i}-{j}: No good fit')
                    continue

                # 9.5. Check fitness w.r.t source points
                fit_p1, fit_p2, fit_p3 = fit
                fit_p1 = fit_p1.reshape(1, spatial_dims_n, 1)
                fit_p2 = fit_p2.reshape(1, spatial_dims_n, 1)
                fit_p3 = fit_p3.reshape(1, spatial_dims_n, 1)

                source_points_of_both = source_points[i] + source_points[j]
                sp_tensor = torch.tensor(source_points_of_both).permute(1, 0).contiguous()
                dists, _ = calculate_canonical_coordinates_with_cardano(fit_p1, fit_p2, fit_p3, sp_tensor,
                                                                        real_distance=True)
                if dists.pow(2).mean().sqrt() > fit_tol:
                    log_debug(f'{i}-{j}: No good fit w.r.t source points')
                    continue
                del sp_tensor, dists, _, fit_p1, fit_p2, fit_p3

                # 10. If the fit is good
                log_debug(f'{i}-{j}: Found a good fit')
                #     remove that curve
                del curves[j]
                del source_points[j]

                #     and replace this curve with the fit
                p1, p2, p3 = fit
                w = (w + that_w) / 2
                tb, pb = bisector_point(p1, p2, p3)
                curves[i] = torch.cat([p1, p2, p3, w[None]])
                source_points[i] = source_points_of_both

        if previous_curves_n == len(curves):
            break
        previous_curves_n = len(curves)
        maxi = max(curves.keys())

    return torch.stack(list(curves.values()))


# def join_quad_beziers(curves, join_tol=.5, fit_tol=.5, w_tol=.5):
#     r"""
#
#     Parameters
#     ----------
#     curves : array_like
#         of shape [curves_n, control_points_n * spatial_dims_n + 1]
#
#     Returns
#     -------
#     curves : torch.Tensor
#         of shape [new_curves_n, control_points_n * spatial_dims_n + 1]
#     """
#     previous_curves_n = len(curves)
#     while True:
#         curves = join_quad_beziers_iter(curves, join_tol, fit_tol, w_tol)
#         if previous_curves_n == len(curves):
#             break
#         previous_curves_n = len(curves)
#
#     return curves
#
#
#
# def join_quad_beziers_iter(curves, join_tol=.5, fit_tol=.5, w_tol=.5):
#     r"""
#
#     Parameters
#     ----------
#     curves : array_like
#         of shape [curves_n, control_points_n * spatial_dims_n + 1]
#
#     Returns
#     -------
#     curves : torch.Tensor
#         of shape [new_curves_n, control_points_n * spatial_dims_n + 1]
#     """
#     initial_curves_n = len(curves)
#     spatial_dims_n = 2
#     curves = {i: torch.as_tensor(curves[i]) for i in range(initial_curves_n)}
#
#     # 1. For every this curve
#     for i in range(initial_curves_n):
#         if i not in curves:
#             continue
#
#         curve = curves[i]
#         p1 = curve[:spatial_dims_n]
#         p2 = curve[spatial_dims_n:spatial_dims_n * 2]
#         p3 = curve[spatial_dims_n * 2:spatial_dims_n * 3]
#         w = curve[spatial_dims_n * 3]
#         tb, pb = bisector_point(p1, p2, p3)
#
#         # 2. For every that curve
#         for j in range(initial_curves_n):
#             if (j not in curves) or (j == i):
#                 continue
#             that_curve = curves[j]
#             that_w = that_curve[spatial_dims_n * 3]
#
#             # 3. If widths are different -- don't fuse
#             if (that_w - w).abs() > w_tol:
#                 log_debug(f'{i}-{j}: Widths are different')
#                 continue
#
#             q1 = that_curve[:spatial_dims_n]
#             q2 = that_curve[spatial_dims_n:spatial_dims_n * 2]
#             q3 = that_curve[spatial_dims_n * 2:spatial_dims_n * 3]
#             sb, qb = bisector_point(q1, q2, q3)
#             del q2
#
#             # 4. Calculate distances to this curve from q1, qb, q3 and their ts
#             p1 = p1.reshape(1, spatial_dims_n, 1)
#             p2 = p2.reshape(1, spatial_dims_n, 1)
#             p3 = p3.reshape(1, spatial_dims_n, 1)
#             coords_q = torch.stack([q1, qb, q3], dim=1)
#             dq, tq = calculate_canonical_coordinates_with_cardano(p1, p2, p3, coords_q, real_distance=True)
#             del coords_q
#             p1 = p1.reshape(spatial_dims_n)
#             p2 = p2.reshape(spatial_dims_n)
#             p3 = p3.reshape(spatial_dims_n)
#             dq1, dqb, dq3 = dq.reshape(3)
#             tq1, tqb, tq3 = tq.reshape(3)
#             del tq, tqb
#
#             # 5. If all three are close -- fuse that curve into this curve
#             if (dq <= join_tol).all():
#                 log_debug(f'{i}-{j}: All three are close')
#                 del curves[j]
#                 continue
#
#             # 6. If all far, or
#             #       ends are close but bisector point is far, or vice versa -- skip
#             if ((dq > join_tol).all() or
#                     ((dq1 <= join_tol) and (dq3 <= join_tol) and (dqb > join_tol)) or
#                     ((dq1 > join_tol) and (dq3 > join_tol) and (dqb <= join_tol))):
#                 log_debug(f'{i}-{j}: Won\'t merge')
#                 continue
#             del dq
#
#             #    Else, either q1 or q3 is far from this curve
#             # 7. Sort the points of that curve so that the closest is q1
#             if dq3 < dq1:
#                 q1, q3 = q3, q1
#                 tq1, tq3 = tq3, tq1
#                 sb = 1 - sb
#             del dq1, dq3
#
#             # 8. Sort the points of this curve so that the points go from p1 to q3, i.e tq3 > 1
#             if tq3 < 0:
#                 p1, p3 = p3, p1
#                 tb = 1 - tb
#                 tq1 = 1 - tq1
#                 tq3 = 1 - tq3
#
#             # 8.5.
#             eps = .05
#             if (tq1 < eps) or (tq3 < 1 + eps):
#                 log_debug(f'{i}-{j}: Won\'t merge')
#                 continue
#             del tq3
#
#             # 9. Find best fit
#             fit, error = find_qbezier_best_join_fit(p1, pb, p3, q1, qb, q3, tb, sb, tq1)
#             del q1, qb, q3, sb, tq1
#             if error > fit_tol:
#                 log_debug(f'{i}-{j}: No good fit')
#                 continue
#
#             # 10. If the fit is good
#             log_debug(f'{i}-{j}: Found a good fit')
#             #     remove that curve
#             del curves[j]
#
#             #     and replace this curve with the fit
#             p1, p2, p3 = fit
#             w = (w + that_w) / 2
#             tb, pb = bisector_point(p1, p2, p3)
#             curves[i] = torch.cat([p1, p2, p3, w[None]])
#
#     return torch.stack(list(curves.values()))


def find_qbezier_best_join_fit(p1, pb, p3, q1, qb, q3, tb, sb, tq1, tries_n=100):
    best_fit = None
    best_error = np.inf
    for uq1 in torch.linspace(0, 1, tries_n + 2):
        fit, error = find_qbezier_join_fit(p1, pb, p3, q1, qb, q3, tb, sb, tq1, uq1)
        if error < best_error:
            best_error = error
            best_fit = fit
    return best_fit, best_error


def find_qbezier_join_fit(p1, pb, p3, q1, qb, q3, tb, sb, tq1, uq1):
    u = p1.new_tensor([0, tb * uq1 / tq1, uq1 / tq1, uq1, 1 - (1 - sb) * (1 - uq1), 1])
    return find_qbezier_fit(torch.stack([p1, pb, p3, q1, qb, q3]), u)


def find_qbezier_fit(pts, t):
    r"""

    Parameters
    ----------
    pts : torch.Tensor
        of shape [pts_n, spatial_dims_n]
    t : torch.Tensor
        of shape [pts_n]

    Returns
    -------
    control_points : torch.Tensor
        of shape [3, spatial_dims_n]
    error : scalar
        MSE of fit
    """
    T = torch.stack([torch.ones_like(t), t, t.pow(2)]).permute(1, 0)
    del t
    M = T.new_tensor([[1, 0, 0], [-2, 2, 0], [1, -2, 1]])
    M = T.matmul(M)
    del T

    try:
        control_points, _ = torch.lstsq(pts, M)
        del pts, M, _
        error = control_points[3:].pow(2).mean().sqrt().item()
        control_points = control_points[:3]
        return control_points, error
    except RuntimeError:
        return None, np.inf


def qbezier_point(p1, p2, p3, t):
    omt = 1 - t
    return p1 * omt.pow(2) + p2 * omt * t * 2 + p3 * t.pow(2)


def bisector_point(p1, p2, p3):
    _ = (p2 - p1).norm().sqrt()
    tb = _ / (_ + (p2 - p3).norm().sqrt())
    pb = qbezier_point(p1, p2, p3, tb)
    return tb, pb


def calculate_canonical_coordinates_with_cardano(p1, p2, p3, pixel_coords, tol=1e-3, real_distance=False, division_epsilon=1e-12):
    r"""
    Parameters
    ----------
    p1 : torch.Tensor
        of shape [patches_n, spatial_dims_n, primitives_n]
    p2 : torch.Tensor
        of shape [patches_n, spatial_dims_n, primitives_n]
    p3 : torch.Tensor
        of shape [patches_n, spatial_dims_n, primitives_n]
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

    Notes
    -----
    Based on this https://ru.wikipedia.org/wiki/%D0%A4%D0%BE%D1%80%D0%BC%D1%83%D0%BB%D0%B0_%D0%9A%D0%B0%D1%80%D0%B4%D0%B0%D0%BD%D0%BE
    """
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

    spatial_dim_i = 1
    a = B.pow(2).sum(dim=spatial_dim_i)
    a = torch.max(a, a.new_full([], division_epsilon))
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
    del t, omt, p2

    spatial_dim_i = 2
    dist_to_curve = closest_on_curve.norm(dim=spatial_dim_i)
    dist_to_curve.masked_fill_(torch.isnan(dist_to_curve), np.inf)
    del closest_on_curve
    dist_to_curve, sol_id = dist_to_curve.min(dim=0)

    sol_id = sol_id.reshape(1, patches_n, primitives_n, pixels_n)
    t_of_closest = torch.gather(t_of_closest, dim=0, index=sol_id)[0]
    del sol_id

    if real_distance:
        spatial_dim_i = 1

        left_end_is_closest = t_of_closest < 0
        p1 = p1.reshape(patches_n, spatial_dims_n, primitives_n, pixels_n)
        dist_to_curve = p1.norm(dim=spatial_dim_i).where(left_end_is_closest, dist_to_curve)
        del left_end_is_closest, p1

        right_end_is_closest = t_of_closest > 1
        p3 = p3.reshape(patches_n, spatial_dims_n, primitives_n, pixels_n)
        dist_to_curve = p3.norm(dim=spatial_dim_i).where(right_end_is_closest, dist_to_curve)
        del right_end_is_closest, p3

    return dist_to_curve, t_of_closest