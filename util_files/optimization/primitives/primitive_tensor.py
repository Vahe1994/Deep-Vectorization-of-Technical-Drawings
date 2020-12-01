import cairocffi as cairo
import numpy as np
import torch

from ..parameters import collinearity_beta, division_epsilon, elementary_halfwidth, empty_pixel_tolerance,\
    foreground_pixel_min_shading, min_visible_width, neighbourhood_padding, reinit_initial_length, reinit_initial_width


class PrimitiveTensor:
    def __init__(self, dtype=None, device=None):
        r"""

        Parameters
        ----------
        dtype : torch.dtype, optional
            The desired data type of the parameters and all computations.
            If None, uses a global default see ``torch.set_default_tensor_type()``
        device : torch.device, optional
            The desired device of the raster grid and all computations. If None, uses the current device for `dtype`
        """
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.dtype = dtype
        if device is None:
            device = torch.empty([], dtype=dtype).device
        self.device = device

        self.canonical_parameters = []

        self.patches_n = None
        self.primitives_n = None
        self._canonical_pixel_coords = None
        self._pixel_coords = None
        self._in_neighbourhood = None
        self._q_collinearity = None

        self.size_fixed = True
        self.pos_fixed = True

    import util_files.optimization.energy.gaussian as energy_procedures

    def fix_pos(self):
        if self.pos_fixed:
            return
        self.pos_fixed = True
        self.invalidate_pos_dependent()

    def free_pos(self):
        if not self.pos_fixed:
            return
        self.pos_fixed = False
        self.invalidate_pos_dependent()

    def invalidate_pos_dependent(self):
        return NotImplementedError

    def fix_size(self):
        if self.size_fixed:
            return
        self.size_fixed = True
        self.invalidate_size_dependent()

    def free_size(self):
        if not self.size_fixed:
            return
        self.size_fixed = False
        self.invalidate_size_dependent()

    def get_q_collinearity(self, pixel_coords, q_prim, collinearity_beta=collinearity_beta,
                           division_epsilon=division_epsilon):
        r"""

        Parameters
        ----------
        pixel_coords : torch.Tensor
            of shape [spatial_dims_n, pixels_n]
        q_prim : torch.Tensor
            of shape [patches_n, primitives_n, pixels_n]

        Returns
        -------
        in_neighbourhood : torch.Tensor
            Tensor of shape [patches_n, primitives_n, pixels_n] with True for pixels in neighbourhood of the primitive.
        """
        # 0. Reuse cached values if they weren't invalidated and pixel_coords are the same
        reuse_pixel_coords = ((self._canonical_pixel_coords is not None) and
                              torch.all(self._pixel_coords == pixel_coords))
        if reuse_pixel_coords and (self._q_collinearity is not None):
            return self._q_collinearity

        patches_n, primitives_n, pixels_n = q_prim.shape

        # 1. Find canonical coordinates of the pixels
        if not reuse_pixel_coords:
            # FIXME resolve the cashing clash with neighbourhood function where from_midpoint=True
            assert False, 'Should not use from_midpoint=True here'
            canonical_x_abs, canonical_y = self.calculate_canonical_coordinates(pixel_coords, from_midpoint=True)
            canonical_x_abs = canonical_x_abs.data
            canonical_y = canonical_y.data
            canonical_x_abs.abs_()
            self._canonical_pixel_coords = canonical_x_abs, canonical_y
            self._pixel_coords = pixel_coords
        del pixel_coords
        canonical_x_abs, canonical_y = self._canonical_pixel_coords

        # 2. Calculate vector field
        q_prim = q_prim.reshape(patches_n, 1, primitives_n, pixels_n)
        vector_field = self.get_vector_field_at(canonical_y) * q_prim
        del canonical_y

        # 3. Calculate vector field of others
        primitives_dim_i = 2
        others_vector_field = vector_field.sum(dim=primitives_dim_i, keepdim=True) - vector_field

        # 4. Calculate collinearity factors
        spatial_dim_i = 1
        others_vf_norm = others_vector_field.norm(dim=spatial_dim_i, keepdim=True)
        others_vector_field /= (others_vf_norm.expand_as(others_vector_field) + division_epsilon)
        collinearity = (others_vector_field * vector_field).sum(spatial_dim_i)
        del vector_field, others_vector_field

        others_vf_norm = others_vf_norm.reshape(patches_n, primitives_n, pixels_n)
        collinearity = torch.exp(-(collinearity.abs() - 1).pow(2) * collinearity_beta) * others_vf_norm
        del others_vf_norm

        self._q_collinearity = collinearity
        return collinearity

    def get_neighbourhood_weighting(self, pixel_coords, pixel_values, empty_pixel_tol=empty_pixel_tolerance,
                                    elementary_halfwidth=elementary_halfwidth, x_padding=neighbourhood_padding,
                                    y_padding=neighbourhood_padding):
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
            Tensor of shape [patches_n, primitives_n, pixels_n] with True for pixels in neighbourhood of the primitive.
        """
        # 0. Reuse cached values if they weren't invalidated and pixel_coords are the same
        reuse_pixel_coords = ((self._canonical_pixel_coords is not None) and
                              torch.all(self._pixel_coords == pixel_coords))
        if reuse_pixel_coords and (self._in_neighbourhood is not None):
            return self._in_neighbourhood

        patches_n = self.patches_n
        primitives_n = self.primitives_n
        spatial_dims_n, pixels_n = pixel_coords.shape

        # 1. Find canonical coordinates of the pixels
        if not reuse_pixel_coords:
            canonical_x_abs, canonical_y = self.calculate_canonical_coordinates(pixel_coords, from_midpoint=True)
            canonical_x_abs = canonical_x_abs.data
            canonical_y = canonical_y.data
            canonical_x_abs.abs_()
            self._canonical_pixel_coords = canonical_x_abs, canonical_y
            self._pixel_coords = pixel_coords
        del pixel_coords
        canonical_x_abs, canonical_y = self._canonical_pixel_coords

        # 2. Select empty raster pixels
        empty = pixel_values <= empty_pixel_tol
        empty = empty.reshape(patches_n, 1, pixels_n).expand(patches_n, primitives_n, pixels_n)
        del pixel_values
        # and keep only pixels within elementary halfwidth
        within_elementary_halfwidth = canonical_x_abs <= elementary_halfwidth
        within_elementary_halfwidth_and_empty = within_elementary_halfwidth & empty

        # 3.1. Among the pixels within elementary halfwidth, select the pixels to the right
        pixels_to_the_right = canonical_y >= 0
        candidates_one_side = pixels_to_the_right & within_elementary_halfwidth_and_empty

        # 3.2. Among these find the leftmost
        pixels_dim_i = 2
        max_y = canonical_y.masked_fill(~candidates_one_side, np.inf)
        del candidates_one_side
        max_y = max_y.min(dim=pixels_dim_i)[0]

        # 3.3. If there is no candidates to the right, i.e the raster is shaded all the way to the patch edge,
        #      use the coordinates of the pixel next to the edge
        backup_candidates = pixels_to_the_right & within_elementary_halfwidth
        max_y_backup = canonical_y.masked_fill(~backup_candidates, -np.inf)
        del backup_candidates
        max_y_backup = max_y_backup.max(dim=pixels_dim_i)[0]
        max_y = max_y.where(torch.isfinite(max_y), max_y_backup)
        del max_y_backup

        # 3.4. If there is no any pixels to the right, i.e the primitive is corrupted somehow,
        #      use the minimal possible value, i.e 0
        max_y.masked_fill_(~torch.isfinite(max_y), 0)

        # 4.1. Among the pixels within elementary halfwidth, select the pixels to the left
        pixels_to_the_left = ~pixels_to_the_right
        del pixels_to_the_right
        candidates_one_side = pixels_to_the_left & within_elementary_halfwidth_and_empty
        del within_elementary_halfwidth_and_empty

        # 4.2. Among these find the rightmost
        min_y = canonical_y.masked_fill(~candidates_one_side, -np.inf)
        del candidates_one_side
        min_y = min_y.max(dim=pixels_dim_i)[0]

        # 4.3. If there is no candidates to the left, i.e the raster is shaded all the way to the patch edge,
        #      use the coordinates of the pixel next to the edge
        backup_candidates = pixels_to_the_left & within_elementary_halfwidth
        del within_elementary_halfwidth, pixels_to_the_left
        min_y_backup = canonical_y.masked_fill(~backup_candidates, +np.inf)
        del backup_candidates
        min_y_backup = min_y_backup.min(dim=pixels_dim_i)[0]
        min_y = min_y.where(torch.isfinite(min_y), min_y_backup)
        del min_y_backup

        # 4.4. If there is no any pixels to the left, i.e the primitive is corrupted somehow,
        #      use the maximal possible value, i.e 0
        min_y.masked_fill_(~torch.isfinite(min_y), 0)

        #    We found the bounds w.r.t canonical y coordinate
        #    Within these bounds find the largest 'rectangle' with only shaded raster pixels, i.e
        # 5.1 Select the empty raster pixels within these bounds
        min_y = min_y.reshape(patches_n, primitives_n, 1)
        max_y = max_y.reshape(patches_n, primitives_n, 1)
        empty_within_y_bounds = canonical_y > min_y
        empty_within_y_bounds &= canonical_y < max_y
        empty_within_y_bounds &= empty

        # 5.2 And find the pixels with minimal absolute x coordinate
        max_x = canonical_x_abs.masked_fill(~empty_within_y_bounds, np.inf)
        del empty_within_y_bounds
        max_x = max_x.min(dim=-1)[0]
        max_x.masked_fill_(~torch.isfinite(max_x), 0)

        # 6. These bounds on y and x coordinates define the neighbourhood. Add some padding
        max_x = max_x.reshape(patches_n, primitives_n, 1)
        in_neighbourhood = ~empty
        del empty
        in_neighbourhood &= canonical_y <= max_y + y_padding
        in_neighbourhood &= canonical_y >= min_y - y_padding
        in_neighbourhood &= canonical_x_abs <= max_x + x_padding
        del max_y, min_y, max_x

        # maybe FIXME: pos energy from v1 also does this. Let's see how it works without this
        # in_neighbourhood &= excess_raster < 0

        # # draw for debugging
        # _ = in_neighbourhood[:, 0]
        # h = int(np.sqrt(pixels_n))
        # assert pixels_n % h == 0
        # w = pixels_n // h
        # _ = _.cpu()
        # _ = _.reshape(patches_n, h, w)
        # self.debug_ax.imshow(np.vstack(_))

        self._in_neighbourhood = in_neighbourhood
        return in_neighbourhood

    def invalidate_size_dependent(self):
        return NotImplementedError

    def reinit_collapsed_primitives(self, pixel_coords, raster, primitive_rasterization,
                                    min_visible_width=min_visible_width, min_foreground=foreground_pixel_min_shading,
                                    initial_width=reinit_initial_width, initial_length=reinit_initial_length):
        r"""

        Parameters
        ----------
        pixel_coords : torch.Tensor
            of shape [spatial_dims_n, pixels_n]
        raster : torch.Tensor
            of shape [patches_n, pixels_n]
        primitive_rasterization : torch.Tensor
            of shape [patches_n, pixels_n]
        """
        patches_n = self.patches_n
        primitives_n = self.primitives_n
        spatial_dims_n, pixels_n = pixel_coords.shape

        # 1. Find a single thinest primitive
        primitives_dim_i = 1
        width = self._width.data.reshape(patches_n, primitives_n)
        thinest_width_in_patch, thinest_id = width.min(dim=primitives_dim_i, keepdim=True)
        del width
        primitives_to_reinit = thinest_width_in_patch.new_zeros(patches_n, primitives_n, dtype=torch.bool)
        primitives_to_reinit.scatter_(dim=primitives_dim_i, index=thinest_id, value=1)
        del thinest_id

        # 2. Mark the patches with collapsed primitives
        patches_with_primitives_to_reinit = thinest_width_in_patch < min_visible_width
        del thinest_width_in_patch

        # 3. In each patch among the pixels not already covered by some primitive,
        #    find the 'most not covered' pixel, i.e with the max shading value
        pixels_dim_i = 1
        not_covered = primitive_rasterization == 0.
        raster = raster.where(not_covered, raster.new_zeros([]))
        del not_covered
        values, pixel_ids = raster.max(dim=pixels_dim_i)
        del raster

        # 4. If the found pixel is of background, then there is nowhere to put the reinitialized primitive
        patches_with_uncovered_pixels = values >= min_foreground
        del values

        # 5. Find patches that have both primitives to reinitialize and pixels to cover
        patches_to_reinit = patches_with_primitives_to_reinit.reshape(patches_n)
        del patches_with_primitives_to_reinit
        patches_to_reinit &= patches_with_uncovered_pixels.reshape(patches_n)
        del patches_with_uncovered_pixels

        if not patches_to_reinit.any():
            return

        # 6. Reinitialize primitives
        primitives_to_reinit &= patches_to_reinit.reshape(patches_n, 1).expand(patches_n, primitives_n)
        pixel_ids = pixel_ids[patches_to_reinit]
        del patches_to_reinit

        pixels_dim_i = 2
        pixel_ids = pixel_ids.reshape(-1, 1, 1).expand(-1, spatial_dims_n, 1)
        pixel_coords = pixel_coords.reshape(1, spatial_dims_n, pixels_n).expand(len(pixel_ids), -1, -1)
        pixel_coords_to_reinit_at = torch.gather(pixel_coords, dim=pixels_dim_i, index=pixel_ids)[:, :, 0]
        del pixel_ids, pixel_coords

        self.reinit_primitives(primitives_to_reinit, pixel_coords_to_reinit_at,
                               initial_width=initial_width, initial_length=initial_length)

    def render_single_primitive_with_cairo(self, ctx, patch_i, primitive_i):
        raise NotImplementedError

    def render_with_cairo_each(self, width, height):
        r"""

        Parameters
        ----------
        width : int
            width of the patch
        height : int
            height of the patch

        Returns
        -------
        renderings : torch.Tensor
            of shape [patches_n, primitives_n, height, width]

        """
        # prepare data buffer to render each group (patch) to
        buffer_width = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, width)
        buffer = torch.empty((height, buffer_width), dtype=torch.uint8).reshape(-1).numpy()

        renderings = torch.empty([self.patches_n, self.primitives_n, height, width], dtype=self.dtype)

        # prepare canvas
        surface = cairo.ImageSurface(cairo.FORMAT_A8, width, height, data=memoryview(buffer), stride=buffer_width)
        with cairo.Context(surface) as ctx:
            ctx.set_operator(cairo.OPERATOR_SOURCE)
            ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
            ctx.set_line_cap(cairo.LINE_CAP_BUTT)

            for patch_i in range(self.patches_n):
                for primitive_i in range(self.primitives_n):
                    ctx.set_source_rgba(0, 0, 0, 0)
                    ctx.paint()  # paint emptyness everywhere
                    ctx.set_source_rgba(0, 0, 0, 1)

                    # draw
                    self.render_single_primitive_with_cairo(ctx, patch_i, primitive_i)

                    # remove int32 padding and copy data
                    renderings[patch_i, primitive_i].numpy()[:] = torch.as_tensor(
                        buffer.reshape(height, buffer_width)[:, :width], dtype=self.dtype) / 255

        return renderings

    def render_with_cairo_total(self, width, height, min_width=0):
        r"""

        Parameters
        ----------
        width : int
            width of the patch
        height : int
            height of the patch

        Returns
        -------
        renderings : torch.Tensor
            of shape [patches_n, height, width]

        """
        # prepare data buffer to render each group (patch) to
        buffer_width = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_A8, width)
        buffer = torch.empty((height, buffer_width), dtype=torch.uint8).reshape(-1).numpy()

        renderings = torch.empty([self.patches_n, height, width], dtype=self.dtype)

        # prepare canvas
        surface = cairo.ImageSurface(cairo.FORMAT_A8, width, height, data=memoryview(buffer), stride=buffer_width)
        with cairo.Context(surface) as ctx:
            ctx.set_operator(cairo.OPERATOR_SOURCE)
            ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
            ctx.set_line_cap(cairo.LINE_CAP_BUTT)

            for patch_i in range(self.patches_n):
                ctx.set_source_rgba(0, 0, 0, 0)
                ctx.paint()  # paint emptyness everywhere
                ctx.set_source_rgba(0, 0, 0, 1)

                # draw
                for primitive_i in range(self.primitives_n):
                    self.render_single_primitive_with_cairo(ctx, patch_i, primitive_i, min_width=min_width)

                # remove int32 padding and copy data
                renderings[patch_i].numpy()[:] = torch.as_tensor(
                    buffer.reshape(height, buffer_width)[:, :width], dtype=self.dtype) / 255

        return renderings

    def step(self):
        pass

    def synchronize_parameters(self):
        self._canonical_pixel_coords = None
        self._in_neighbourhood = None

    def unit_energy(self, pixel_coords):
        r"""

        Parameters
        ----------
        pixel_coords : torch.Tensor
            of shape [spatial_dims_n, pixels_n]

        Returns
        -------
        energies : torch.Tensor
            of shape [patches_n, primitives_n, pixels_n]
        """
        raise NotImplementedError
