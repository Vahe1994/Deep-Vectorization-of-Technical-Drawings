import numpy as np
import torch

point_potential_close_range = 1
point_potential_far_range = 32
point_potential_far_weight = 1./50
point_potential_close_weight = 1 - point_potential_far_weight


def unit_term_point_to_point(r2, term_range):
    r"""

    Parameters
    ----------
    r2 : torch.Tensor
        of any shape. Squared distance between the points
    term_range : scalar

    Returns
    -------
    energies : torch.Tensor
        of same shape as `r2`
    """
    return torch.exp(-r2 / (term_range ** 2))


def unit_energy_point_to_point(r2, close_range=point_potential_close_range, close_weight=point_potential_close_weight,
                               far_range=point_potential_far_range, far_weight=point_potential_far_weight):
    r"""

    Parameters
    ----------
    r2 : torch.Tensor
        of any shape. Squared distance between the points
    far_weight : scalar
    far_range : scalar
    close_weight : scalar
    close_range : scalar

    Returns
    -------
    energies : torch.Tensor
        of same shape as `r2`
    """
    return (unit_term_point_to_point(r2, close_range) * close_weight +
            unit_term_point_to_point(r2, far_range) * far_weight)


def unit_term_line_to_canonical_point(halfwidth, length, x, y, term_range):
    r"""Term of energy of interaction of a line and a point in canonical coordinates of the line,
    where the left end of the line is in the origin and the right end is on the positive side of y axis.

    Parameters `halfwidth`, `length`, `x`, and `y` should be broadcastable.

    Parameters
    ----------
    halfwidth : torch.Tensor
        of any shape
    length : torch.Tensor
        of same shape
    x : torch.Tensor
        of same shape
    y : torch.Tensor
        of same shape
    term_range : scalar

    Returns
    -------
    energies : torch.Tensor
        of same shape as input
    """
    erf = lambda arg: torch.erf(arg / term_range)
    return (erf(length - y) + erf(y)) * (erf(halfwidth - x) + erf(halfwidth + x)) * (term_range ** 2) * np.pi / 4


def unit_energy_line_to_canonical_point(halfwidth, length, x, y, close_range=point_potential_close_range,
                                        close_weight=point_potential_close_weight, far_range=point_potential_far_range,
                                        far_weight=point_potential_far_weight):
    r"""Energy of interaction of a line and a point in canonical coordinates of the line,
    where the left end of the line is in the origin and the right end is on the positive side of y axis.

    Parameters `halfwidth`, `length`, `x`, and `y` should be broadcastable.

    Parameters
    ----------
    halfwidth : torch.Tensor
        of any shape
    length : torch.Tensor
        of same shape
    x : torch.Tensor
        of same shape
    y : torch.Tensor
        of same shape
    far_weight : scalar
    far_range : scalar
    close_weight : scalar
    close_range : scalar

    Returns
    -------
    energies : torch.Tensor
        of same shape as input
    """
    return (unit_term_line_to_canonical_point(halfwidth, length, x, y, close_range) * close_weight +
            unit_term_line_to_canonical_point(halfwidth, length, x, y, far_range) * far_weight)

