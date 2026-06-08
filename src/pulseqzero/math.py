import torch
import numpy as np


class Ceil(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ceil(x)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs


class Floor(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.floor(x)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs


class Round(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.round(x)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs


def ceil(x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of torch.ceil.
    For gradient calculation, this mimicks the identity function."""
    try:
        return Ceil.apply(x)
    except TypeError:
        return torch.as_tensor(np.ceil(x))


def floor(x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of torch.floor.
    For gradient calculation, this mimicks the identity function."""
    try:
        return Floor.apply(x)
    except TypeError:
        return torch.as_tensor(np.floor(x))


def round(x: torch.Tensor) -> torch.Tensor:
    """Differentiable version of torch.round.
    For gradient calculation, this mimicks the identity function."""
    try:
        return Round.apply(x)
    except TypeError:
        return torch.as_tensor(np.round(x))
    
def interp(x, xp, fp, left=None, right=None, tol=None) -> torch.Tensor:
    """Autograd-compatible 1D linear interpolation, mirroring numpy.interp.

    Linear within ``[xp[0], xp[-1]]``; outside it returns ``left`` (for
    ``x < xp[0]``) and ``right`` (for ``x > xp[-1]``), defaulting to the edge
    values ``fp[0]`` / ``fp[-1]`` like numpy. Pass ``left=right=0`` to zero-fill
    (a gradient that is not playing outside its support).

    ``tol`` is the slack on the support test (beyond numpy): the default few-ULP
    slack lets a query landing exactly on a boundary survive float rounding (e.g.
    when ``xp`` and ``x`` are computed in different dtypes); pass ``tol=0`` for a
    strict boundary when the query grid is built from the exact ``xp``.

    Differentiable in x, xp and fp. ``xp`` must be sorted ascending.
    """
    x, xp, fp = torch.as_tensor(x), torch.as_tensor(xp), torch.as_tensor(fp)
    xp = xp.to(x.dtype)
    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[:-1] - m * xp[:-1]  # offset

    idx = (torch.searchsorted(xp, x, right=False) - 1).clamp(0, xp.numel() - 2)
    y = m[idx] * x + b[idx]

    if tol is None:
        tol = 8 * torch.finfo(x.dtype).eps * xp.abs().amax()  # << raster, >> noise
    left = fp[0] if left is None else left
    right = fp[-1] if right is None else right
    y = torch.where(x < xp[0] - tol, torch.as_tensor(left, dtype=y.dtype), y)
    y = torch.where(x > xp[-1] + tol, torch.as_tensor(right, dtype=y.dtype), y)
    return y

def round_half_up(n, decimals=0):
    """Differentiable version of rouding (avoiding banker's rounding).
    
    BUG: will strip signs - same as pypulseq impl!
    """
    # Pass trhough on backwards pass - using pulseq-zeros floor impl
    multiplier = 10 ** decimals
    return floor(abs(n) * multiplier + 0.5) / multiplier
