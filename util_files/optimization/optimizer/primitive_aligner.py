import numpy as np
import matplotlib.pyplot as plt
import torch

from .logging import Logger
from ..parameters import neighbourhood_pos_weight, pixel_center_coodinates_are_integer, reinit_period


class PrimitiveAligner:
    def __init__(self, initial_primitives_tensor, raster_tensor, logger=None, loglevel='debug'):
        r"""

        Parameters
        ----------
        initial_primitives_tensor : PrimitiveTensor
        raster_tensor : torch.Tensor
            of shape [patches_n, patch_height, patch_width]
        loglevel
        """
        if logger is None:
            self.logger = Logger.prepare_logger(loglevel=loglevel, logfile=None)
        else:
            self.logger = logger
        self.prim_ten = initial_primitives_tensor
        dtype = self.prim_ten.dtype
        device = self.prim_ten.device

        raster_tensor = raster_tensor.type(dtype)
        self.q_raster = raster_tensor.to(device)
        self.pixel_coords = prepare_pixel_coordinates(raster_tensor).to(device)

        patches_n = raster_tensor.shape[0]

    def prepare_visualization(self, patch_ids=None, ax_size=5, fontsize=None, with_debug_ax=False, store_plots=False):
        if patch_ids is not None:
            self.patch_ids = patch_ids
        else:
            patch_ids = self.patch_ids = list(range(self.prim_ten.patches_n))
        plots_n = 6 if not with_debug_ax else 7
        fig, axes = plt.subplots(1, plots_n, figsize=[ax_size * plots_n, ax_size * len(patch_ids)])
        if not with_debug_ax:
            initial_pred_ax, initial_skeleton_ax, refined_pred_ax, refined_skeleton_ax, target_ax, dif_ax = axes
        else:
            initial_pred_ax, initial_skeleton_ax, refined_pred_ax, refined_skeleton_ax, target_ax, dif_ax = axes[1:]
            self.prim_ten.debug_ax = axes[0]

        if fontsize is None:
            fontsize = 3 * ax_size
        initial_pred_ax.set_xlabel('Initial prediction', fontsize=fontsize)
        initial_skeleton_ax.set_xlabel('Initial skeleton', fontsize=fontsize)
        refined_pred_ax.set_xlabel('Refined prediction', fontsize=fontsize)
        refined_skeleton_ax.set_xlabel('Refined skeleton', fontsize=fontsize)
        target_ax.set_xlabel('Target', fontsize=fontsize)
        dif_ax.set_xlabel('Difference', fontsize=fontsize)

        for ax in axes:
            ax.xaxis.set_label_position('top')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        fig.subplots_adjust(wspace=0, hspace=0)

        # draw initials
        _, image_height, image_width = self.q_raster.shape

        self.im_target = im_target = np.vstack(self.q_raster.cpu()[patch_ids])
        self.target_plot = target_ax.imshow(im_target, vmin=0, vmax=1, cmap='gray_r')

        initial_prediction = np.vstack(self.prim_ten.render_with_cairo_total(image_width, image_height)[patch_ids])
        self.initial_pred_plot = initial_pred_ax.imshow(initial_prediction, vmin=0, vmax=1, cmap='gray_r')

        initial_skeleton = self.prim_ten.render_skeleton_total(image_width, image_height)[patch_ids]
        self.initial_skel_plot = initial_skeleton_ax.imshow(np.vstack(initial_skeleton))

        refined_prediction = initial_prediction.copy()
        self.refined_pred_plot = refined_pred_ax.imshow(refined_prediction, vmin=0, vmax=1, cmap='gray_r')

        refined_skeleton = initial_skeleton.copy()
        self.refined_skel_plot = refined_skeleton_ax.imshow(np.vstack(refined_skeleton))

        difference = im_target - initial_prediction
        self.dif_plot = dif_ax.imshow(difference, vmin=-1, vmax=1, cmap='gray_r')

        if store_plots:
            self.stored_plots = []
            self.fig = fig
        else:
            self.stored_plots = None

        return fig

    def draw_visualization(self, patch_rasterization):
        patch_ids = self.patch_ids
        im_refined = np.vstack(patch_rasterization[patch_ids])
        self.refined_pred_plot.set_array(im_refined)

        im_dif = im_refined - self.im_target
        self.dif_plot.set_array(im_dif)

        _, image_height, image_width = self.q_raster.shape
        refined_skeleton = np.vstack(self.prim_ten.render_skeleton_total(image_width, image_height)[patch_ids])
        self.refined_skel_plot.set_array(refined_skeleton)

        if self.stored_plots is not None:
            self.stored_plots.append([im_refined, im_dif, refined_skeleton])

    def save_plots(self, filepath, fps=30):
        if self.stored_plots is None:
            raise ValueError('Not prepared for saving')

        import matplotlib.animation as animation

        def animate_func(frame_i):
            im_refined, im_dif, im_refined_skeleton = self.stored_plots[frame_i]
            self.refined_pred_plot.set_array(im_refined)
            self.dif_plot.set_array(im_dif)
            self.refined_skel_plot.set_array(im_refined_skeleton)
            return [self.initial_pred_plot, self.initial_skel_plot, self.refined_pred_plot,
                    self.refined_skel_plot, self.target_plot, self.dif_plot]

        anim = animation.FuncAnimation(self.fig, animate_func, frames=len(self.stored_plots))
        return anim.save(filepath, fps=fps, extra_args=['-vcodec', 'libx264'])

    def step(self, iteration_i, draw_visualization=False, reinit_period=reinit_period):
        logger = self.logger
        prim_ten = self.prim_ten
        q_raster = self.q_raster
        pixel_coords = self.pixel_coords

        dtype = prim_ten.dtype
        device = prim_ten.device
        primitives_n = prim_ten.primitives_n

        self.zero_grad()

        logger.debug('Calculate renderings')  # %%
        patches_n, patch_height, patch_width = q_raster.shape
        pixels_n = patch_height * patch_width
        q_prim = prim_ten.render_with_cairo_each(patch_width, patch_height).to(device, dtype)  # patches_n x prims_n x height x width
        primitive_rasterization = prim_ten.render_with_cairo_total(patch_width, patch_height)  # used later for visualization
        q_all = primitive_rasterization.to(device, dtype)  # patches_n x height x width

        logger.debug('Reinitialize')  # %%
        if iteration_i % reinit_period == 0:
            prim_ten.reinit_collapsed_primitives(pixel_coords, q_raster.reshape(patches_n, pixels_n),
                                                 q_all.reshape(patches_n, pixels_n))
            prim_ten.synchronize_parameters()

        if draw_visualization:  # %%
            logger.debug('Draw visualization')
            self.draw_visualization(primitive_rasterization)
        del primitive_rasterization

        logger.debug('Calculate neighbourhoods')  # %%
        c_prim = prim_ten.get_neighbourhood_weighting(pixel_coords, q_raster.reshape(patches_n, pixels_n))  # patches_n x prims_n x pixels_n

        logger.debug('Calculate pos energies')  # %%
        # same as: q_pos = (q_all - q_prim - q_raster) * (1 + c_prim * (neighbourhood_pos_weight - 1))
        q_pos = q_all.unsqueeze(1) - q_prim
        q_pos -= q_raster.unsqueeze(1)
        q_pos = q_pos.reshape(patches_n, primitives_n, pixels_n)
        q_pos = (q_pos * neighbourhood_pos_weight).where(c_prim, q_pos)

        prim_ten.fix_size()
        prim_ten.free_pos()
        energy_pos = prim_ten.unit_energy(pixel_coords) * q_pos
        del q_pos
        # # draw for debugging
        # _ = energy_pos[:, 0].detach()
        # _ = _.cpu()
        # _ = _.reshape(patches_n, patch_height, patch_width)
        # prim_ten.debug_ax.imshow(np.vstack(_))
        energy_pos = energy_pos.sum(dim=[1, 2]).mean()
        energy_pos.backward()
        logger.debug(f'Pos energy {energy_pos.item()}')
        del energy_pos

        logger.debug('Calculate size energies')  # %%
        # same as: q_size = torch.where(c_prim, q_all - q_raster, q_prim)
        q_size = q_all
        del q_all
        q_size -= q_raster
        q_prim = q_prim.reshape(patches_n, primitives_n, pixels_n)
        q_size = q_size.reshape(patches_n, 1, pixels_n).expand(patches_n, primitives_n, pixels_n)
        q_size = q_size.where(c_prim, q_prim)
        del c_prim

        logger.debug('Calculate collinearity energies')  # %%
        q_collinearity = prim_ten.get_q_collinearity(pixel_coords, q_prim)
        del q_prim
        q_size += q_collinearity
        del q_collinearity

        prim_ten.fix_pos()
        prim_ten.free_size()
        energy_size = prim_ten.unit_energy(pixel_coords) * q_size
        del q_size
        energy_size = energy_size.sum(dim=[1, 2]).mean()
        energy_size.backward()
        logger.debug(f'Size energy {energy_size.item()}')
        del energy_size

        self.optimization_step()
        prim_ten.synchronize_parameters()

        logger.debug('Join lined up primitives')  # %%
        # TODO

        logger.debug('Apply constraints')  # %%
        prim_ten.constrain_parameters(patch_width=patch_width, patch_height=patch_height)
        prim_ten.synchronize_parameters()

    def optimization_step(self):
        for p in self.prim_ten.canonical_parameters:
            p = p['parameter']
            if p.grad is not None:
                grad = p.grad.data
                grad[~torch.isfinite(grad)] = 0

    def zero_grad(self):
        for p in self.prim_ten.canonical_parameters:
            p = p['parameter']
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


def prepare_pixel_coordinates(raster_tensor, dtype=None):
    patch_height, patch_width = raster_tensor.shape[1:]
    if dtype is None:
        dtype = raster_tensor.dtype

    pixel_coords = torch.meshgrid(torch.arange(patch_height, dtype=dtype), torch.arange(patch_width, dtype=dtype))
    pixel_coords = torch.stack([pixel_coords[1].reshape(-1), pixel_coords[0].reshape(-1)])
    if not pixel_center_coodinates_are_integer:
        pixel_coords += .5

    return pixel_coords
