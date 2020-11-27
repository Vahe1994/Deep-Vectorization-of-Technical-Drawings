def calculate_canonical_coordinates(pixel_coords, p1, dir):
    r"""

    Parameters
    ----------
    pixel_coords : torch.Tensor
        of shape [spatial_dims_n, pixels_n]
    p1 : torch.Tensor
        of shape [patches_n, spatial_dims_n, primitives_n]
    dir : torch.Tensor
        of shape [patches_n, spatial_dims_n, primitives_n]

    Returns
    -------
    canonical_x : torch.Tensor
        of shape [patches_n, primitives_n, pixels_n]
    canonical_y : torch.Tensor
        of shape [patches_n, primitives_n, pixels_n]
    """
    spatial_dims_n, pixels_n = pixel_coords.shape
    patches_n, _, primitives_n = p1.shape

    pixel_coords = pixel_coords.reshape(1, spatial_dims_n, 1, pixels_n)
    p1 = p1.reshape(patches_n, spatial_dims_n, primitives_n, 1)
    dir = dir.reshape(patches_n, spatial_dims_n, primitives_n, 1)
    translated_pixel_x = pixel_coords[:, 0] - p1[:, 0]  # [patches_n, primitives_n, pixels_n]
    translated_pixel_y = pixel_coords[:, 1] - p1[:, 1]
    del pixel_coords
    canonical_pixel_x = translated_pixel_x * dir[:, 1] - translated_pixel_y * dir[:, 0]
    canonical_pixel_y = translated_pixel_x * dir[:, 0] + translated_pixel_y * dir[:, 1]
    del translated_pixel_x, translated_pixel_y, p1, dir
    return canonical_pixel_x, canonical_pixel_y

