from turtle import right
from copy import copy
from typing import Optional, TypeVar
from pypulseq import Opts
from ..events import TrapGrad, ExtTrapGrad, ArbitraryGrad, Array
import torch
import numpy as np

GradType = TypeVar("GradType", TrapGrad, ExtTrapGrad, ArbitraryGrad)


def scale_grad(grad: GradType, scale: float, system: Optional[Opts] = None) -> GradType:
    grad = copy(grad)

    if isinstance(grad, TrapGrad):
        grad.amplitude = scale * grad.amplitude
    elif isinstance(grad, ExtTrapGrad):
        grad.waveform = scale * grad.waveform
    else:
        grad.waveform = scale * grad.waveform
        grad.first = scale * grad.first
        grad.last = scale * grad.last

    return grad


def points_to_waveform(
    amplitudes: Array, grad_raster_time: float, times: np.ndarray
) -> Array:
    """Only differentiable in amplitude; the time regridding is not."""
    if amplitudes.size == 0:
        return np.zeros(1)

    grid = np.arange(
        round(np.min(times) / grad_raster_time),
        round(np.max(times) / grad_raster_time),
    )
    time_grid = grid * grad_raster_time + grad_raster_time / 2

    if isinstance(amplitudes, torch.Tensor):
        return _torch_interp(x=time_grid, xp=times, fp=amplitudes)
    else:
        return np.interp(x=time_grid, xp=times, fp=amplitudes)


def _torch_interp(x, xp, fp):
    """torch replacement for numpys interp. Differentiable in fp."""
    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[:-1] - (m * xp[:-1])  # offset
    
    indices = torch.searchsorted(xp, x, right=False)
    indices = (indices - 1).clamp(0, len(indices) - 1)

    return m[indices] * x + b[indices]