import torch

from .primitive_aligner import PrimitiveAligner


class Adam(PrimitiveAligner):
    def __init__(self, initial_primitives_tensor, raster_tensor, logger=None, loglevel='debug', lr=1):
        r"""

        Parameters
        ----------
        initial_primitives_tensor : PrimitiveTensor
        raster_tensor : torch.Tensor
            of shape [patches_n, patch_height, patch_width]
        loglevel
        """
        super().__init__(initial_primitives_tensor=initial_primitives_tensor, raster_tensor=raster_tensor,
                         logger=logger, loglevel=loglevel)

        parameter_groups = []
        for p in self.prim_ten.canonical_parameters:
            d = {'params': [p['parameter']]}
            if 'lr_factor' in p:
                d['lr'] = lr * p['lr_factor']
            parameter_groups.append(d)

        self.optimizer = torch.optim.Adam(parameter_groups, lr=lr)
        self.prim_ten.optimizer = self.optimizer

    def optimization_step(self):
        super().optimization_step()
        self.optimizer.step()
        # TODO
