import torch
import numpy as np

def calculate_canonical_coordinates_with_probes(self, pixel_coords):
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
    pixels_n = pixel_coords.shape[1]

    # 1. Sample probes on the curve
    probes, arc_lens, tangents = self.probe()
    probes_n, patches_n, spatial_dims_n, primitives_n = probes.shape
    curve_lengths = arc_lens[-1]

    # 2. For each pixel find the closest probe
    distance_to_curve = probes.new_full([patches_n, primitives_n, pixels_n], np.inf)
    closest_probe_id = torch.empty_like(distance_to_curve, dtype=torch.long)
    # loop over probes instead of batch calculation to loosen up memory requirements
    # no grad from here
    probes = probes.reshape(probes_n, patches_n, spatial_dims_n, primitives_n, 1)
    pixel_coords = pixel_coords.reshape(1, spatial_dims_n, 1, pixels_n)
    spatial_dim_i = 1
    for probe_i in range(probes_n):
        dist_to_this_probe = torch.norm(probes.data[probe_i] - pixel_coords.data, dim=spatial_dim_i)
        it_is_closest_so_far = dist_to_this_probe.data < distance_to_curve.data
        distance_to_curve = dist_to_this_probe.where(it_is_closest_so_far, distance_to_curve)
        closest_probe_id.masked_fill_(it_is_closest_so_far, probe_i)
    del distance_to_curve, it_is_closest_so_far
    # to here

    # 3. Calculate the canonical coordinates of the pixels.
    #    Canonical X is the distance from the pixel to the curve, i.e the distance to the closest probe.
    #    Canonical Y is the length of the curve arc from P1 to the closest probe.
    expanded_transposed_probes = (probes.reshape(probes_n, patches_n, spatial_dims_n, primitives_n, 1)
                                  .permute(1, 2, 3, 4, 0)
                                  .expand(patches_n, spatial_dims_n, primitives_n, pixels_n, probes_n))
    closest_probe_id = (closest_probe_id.reshape(patches_n, 1, primitives_n, pixels_n, 1)
                        .expand(patches_n, spatial_dims_n, primitives_n, pixels_n, 1))
    projections = torch.gather(expanded_transposed_probes, dim=-1, index=closest_probe_id, sparse_grad=False)
    del expanded_transposed_probes
    projections = projections[..., 0]  # get rid of probes dimension
    from_curve_to_pixel = pixel_coords - projections
    del projections, pixel_coords
    spatial_dim_i = 1
    canonical_pixel_x = torch.norm(from_curve_to_pixel, dim=spatial_dim_i)

    tangents = (tangents.reshape(probes_n, patches_n, spatial_dims_n, primitives_n, 1)
                .permute(1, 2, 3, 4, 0)
                .expand(patches_n, spatial_dims_n, primitives_n, pixels_n, probes_n))
    tangents = torch.gather(tangents, dim=-1, index=closest_probe_id, sparse_grad=False)[..., 0]
    x_sign = torch.sign(tangents[:, 1] * from_curve_to_pixel.data[:, 0] -
                        tangents[:, 0] * from_curve_to_pixel.data[:, 1])
    del tangents
    canonical_pixel_x = canonical_pixel_x * x_sign
    del x_sign

    arc_lens = (arc_lens.reshape(probes_n, patches_n, primitives_n, 1)
                .permute(1, 2, 3, 0)
                .expand(patches_n, primitives_n, pixels_n, probes_n))
    closest_probe_id = closest_probe_id[:, 0]  # get rid of spatial dimension
    canonical_pixel_y = torch.gather(arc_lens, dim=-1, index=closest_probe_id, sparse_grad=False)
    del arc_lens
    canonical_pixel_y = canonical_pixel_y[..., 0]  # get rid of probes dimension

    # 4. For pixels for which the closest probes are at the ends of the curves,
    #    redefine the coordinates w.r.t their projection to the tangent line at the end of the curve.
    closest_probe_id = closest_probe_id.reshape(patches_n, 1, primitives_n, pixels_n)
    spatial_dim_i = 1

    def replace_coordinates_of_end_pixels(mask, tangent, y_end_clamper):
        nonlocal canonical_pixel_y, canonical_pixel_x
        # Y
        tangent = tangent.reshape(patches_n, spatial_dims_n, primitives_n, 1)
        from_endpoint_to_pixel = from_curve_to_pixel.where(mask, from_curve_to_pixel.new_zeros([]))
        canonical_pixel_y_end = (tangent[:, 0] * from_endpoint_to_pixel[:, 0] +
                                 tangent[:, 1] * from_endpoint_to_pixel[:, 1])
        canonical_pixel_y_end = y_end_clamper(canonical_pixel_y_end)
        canonical_pixel_y = canonical_pixel_y + canonical_pixel_y_end
        del canonical_pixel_y_end

        # X
        canonical_pixel_x_end = (tangent[:, 1] * from_endpoint_to_pixel[:, 0] -
                                 tangent[:, 0] * from_endpoint_to_pixel[:, 1])
        del tangent, from_endpoint_to_pixel
        # canonical_pixel_x_end = canonical_pixel_x_end.abs()
        mask = mask[:, 0]  # get rid of spatial dimension
        canonical_pixel_x = canonical_pixel_x_end.where(mask, canonical_pixel_x)
        del mask, canonical_pixel_x_end

    right_end_pixels = closest_probe_id == probes_n - 1
    tangent = torch.nn.functional.normalize(self.p3 - self.p2, dim=spatial_dim_i)
    replace_coordinates_of_end_pixels(right_end_pixels, tangent, lambda y_end: torch.clamp(y_end, min=0))
    del right_end_pixels, tangent

    left_end_pixels = closest_probe_id == 0
    del closest_probe_id
    tangent = torch.nn.functional.normalize(self.p2 - self.p1, dim=spatial_dim_i)
    replace_coordinates_of_end_pixels(left_end_pixels, tangent, lambda y_end: torch.clamp(y_end, max=0))
    del left_end_pixels, tangent

    # # draw coordinates for debugging
    # patches_n, primitives_n, pixels_n = canonical_pixel_x.shape
    # h = int(np.sqrt(pixels_n))
    # assert pixels_n % h == 0
    # w = pixels_n // h
    # _ = canonical_pixel_x.detach()[:, 0].cpu()
    # _ = _.reshape(patches_n, h, w)
    # self.debug_ax.imshow(np.vstack(_))

    return canonical_pixel_x, canonical_pixel_y


def probe(self, interval=1):
    r"""Probe curves at intervals not greater than `interval`.

    Parameters
    ----------
    interval

    Returns
    -------
    probes : torch.Tensor
        of shape [probes_n, patches_n, spatial_dims_n, primitives_n]
    arc_lens : torch.Tensor
        of shape [probes_n, patches_n, 1, primitives_n]
    tangets : torch.Tensor
        of shape [probes_n, patches_n, spatial_dims_n, primitives_n]
    """
    t = self.probe_ts(interval=interval)
    t = t.reshape(-1, 1, 1, 1)
    omt = 1 - t

    probes = self.p1 * omt.pow(2) + self.p2 * omt * t * 2 + self.p3 * t.pow(2)
    del t, omt

    spatial_dim_i = 2
    arc_lens = probes[1:] - probes[:-1]

    tangents = torch.nn.functional.normalize(arc_lens.data, dim=spatial_dim_i)
    tangents = torch.cat([tangents, tangents[-1:]], dim=0)

    arc_lens = arc_lens.norm(dim=spatial_dim_i, keepdim=True)
    assert arc_lens.data.max() <= interval, 'We probed the curves at intervals greater than `interval`'
    arc_lens = arc_lens.cumsum(dim=0)
    arc_lens = torch.nn.functional.pad(arc_lens, [0, 0, 0, 0, 0, 0, 1, 0])

    return probes, arc_lens, tangents


def probe_ts(self, interval=1):
    r"""Get t values to probe the curves at intervals not greater than `interval`.

    Returns
    -------
    ts : torch.Tensor
        of shape [probes_n]

    Notes
    -------
    We want ||dB|| to be less than `interval`. At the same time, ||dB|| <= dt * max_{t in [0,1]} ||B'||.
    The derivative of quadratic Bezier curve is
    B' = 2 * ((P2 - P1)(1 - t) + (P3 - P2)t) and is maximal at either t = 0 or 1.
    So we find the max value of derivative across the primitive tensor
    and probe ts uniformly so that dt = `interval` / max_value
    """
    b_prim_max = torch.max(self._p2_to_p1_len.data.max(), self._p2_to_p3_len.data.max()) * 2
    probes_n = int((b_prim_max / interval + 1).ceil())
    ts = torch.linspace(0, 1, probes_n).type(self.dtype).to(self.device)
    return ts
