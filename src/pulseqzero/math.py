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
    
def interp(x, xp, fp) -> torch.Tensor: 
    """ Autograd-compatible 1D linear interpolation. 
    
    x : tensor of points to interpolate 
    xp : 1D tensor of known x points (must be sorted) 
    fp : 1D tensor of known y points 
    """ 
    
    # Ensure xp is sorted 
    assert torch.all(xp[:-1] <= xp[1:]), "xp must be sorted" 
    
    # Expand dimensions for broadcasting 
    x_exp = x.unsqueeze(-1) # shape (..., 1) 
    xp_exp = xp.unsqueeze(0) # shape (1, n) 
    
    # Find the interval: xp[i] <= x < xp[i+1] 
    mask = (xp_exp <= x_exp) # boolean mask 
    idx = mask.sum(dim=-1) - 1 # index of  left point 
    idx = torch.clamp(idx, 0, len(xp)-2) 
    
    x0 = xp[idx] 
    x1 = xp[idx+1] 
    y0 = fp[idx] 
    y1 = fp[idx+1] 
    
    slope = (y1 - y0) / (x1 - x0) 
    y = y0 + slope * (x - x0) 
    
    return y

def round_half_up(n, decimals=0):
    """Differentiable version of rouding (avoiding banker's rounding).
    
    BUG: will strip signs - same as pypulseq impl!
    """
    # Pass trhough on backwards pass - using pulseq-zeros floor impl
    multiplier = 10 ** decimals
    return floor(abs(n) * multiplier + 0.5) / multiplier
