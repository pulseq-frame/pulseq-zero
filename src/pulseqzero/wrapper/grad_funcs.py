from copy import copy
from warnings import warn
from typing import Optional, TypeVar, cast, TypeGuard
from pypulseq import Opts
from ..events import TrapGrad, ExtTrapGrad, ArbitraryGrad, Array, Scalar
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


def split_gradient(
    grad: TrapGrad, system: Optional[Opts] = None
) -> tuple[ExtTrapGrad, ExtTrapGrad, ExtTrapGrad]:
    from .make_grad import make_extended_trapezoid

    if not isinstance(grad, TrapGrad):
        raise ValueError(
            "split_gradient is only implemented for trapezoidal gradients, "
            f"{type(grad)} is not supported."
        )

    def as_tensor(values: list[Scalar]) -> torch.Tensor:
        return torch.stack([torch.as_tensor(v) for v in values])

    if system is None:
        system = Opts.default
    times = as_tensor(
        _cumsum(grad.delay, grad.rise_time, grad.flat_time, grad.fall_time)
    )

    ramp_up = make_extended_trapezoid(
        channel=grad.channel,
        system=system,
        times=times[0:2],
        amplitudes=as_tensor([0, grad.amplitude]),
        skip_check=True,
    )

    flat_top = make_extended_trapezoid(
        channel=grad.channel,
        system=system,
        times=times[1:3],
        amplitudes=as_tensor([grad.amplitude, grad.amplitude]),
        skip_check=True,
    )

    ramp_down = make_extended_trapezoid(
        channel=grad.channel,
        system=system,
        times=times[2:4],
        amplitudes=as_tensor([grad.amplitude, 0]),
        skip_check=True,
    )

    return ramp_up, flat_top, ramp_down


def add_gradients(
    grads: list[TrapGrad | ExtTrapGrad | ArbitraryGrad],
    max_grad: Scalar = 0,
    max_slew: Scalar = 0,
    system: Optional[Opts] = None,
) -> TrapGrad | ExtTrapGrad | ArbitraryGrad:
    from .make_grad import make_trapezoid, make_arbitrary_grad, make_extended_trapezoid

    warn(
        "add_gradients() was written with the help of LLMs; the pypulseq code "
        "is a port of MATLAB code and broken in some circumstances. This code "
        "aims to be a sensible implementation, not a 1:1 replica of pypulseq."
    )

    if len(grads) == 0:
        raise ValueError("No gradients specified")
    if len(grads) == 1:
        # Shallow copy: deepcopy would raise on non-leaf tensor fields (e.g. an
        # amplitude derived from an optimized parameter).
        return copy(grads[0])

    # all gradients must have the same channel
    channel = grads[0].channel
    if channel not in ["x", "y", "z"]:
        raise ValueError(
            f"Invalid channel. Must be one of `x`, `y` or `z`. Passed: {channel}"
        )
    if not all(g.channel == channel for g in grads):
        raise ValueError("Cannot add gradients on different channels.")

    # The upstream PyPulseq routine for oversampled grads is broken.
    if any(isinstance(g, ArbitraryGrad) and g.oversampling for g in grads):
        raise NotImplementedError(
            "add_gradients() does not support oversampled gradients."
        )

    # set defaults
    if system is None:
        system = Opts.default
    if max_grad <= 0:
        max_grad = cast(float, system.max_grad)
    if max_slew <= 0:
        max_slew = cast(float, system.max_slew)

    # =========================================================================
    # Check if we have a set of traps with the same timing
    # =========================================================================
    if (
        _all_traps(grads)
        and all(g.rise_time == grads[0].rise_time for g in grads)
        and all(g.flat_time == grads[0].flat_time for g in grads)
        and all(g.fall_time == grads[0].fall_time for g in grads)
        and all(g.delay == grads[0].delay for g in grads)
    ):
        grad = make_trapezoid(
            grads[0].channel,
            amplitude=sum(g.amplitude for g in grads),
            rise_time=grads[0].rise_time,
            flat_time=grads[0].flat_time,
            fall_time=grads[0].fall_time,
            delay=grads[0].delay,
            system=system,
        )
        return grad

    # =========================================================================
    # Only trapezoids and extended trapezoids (no arbitrary grads): join their
    # piecewise-linear shapes on the union of all control-point times.
    # =========================================================================
    if not any(isinstance(g, ArbitraryGrad) for g in grads):
        shapes = [_control_points(g) for g in grads]

        # Union time axis: sorted, with near-coincident points merged away.
        # (Timing gradients are exact except when two breakpoints exactly
        # coincide, where the merge makes the area non-smooth.)
        times = torch.sort(torch.cat([tt for tt, _ in shapes]))[0]
        keep = torch.cat((times.new_ones(1, dtype=torch.bool), times.diff() > 1e-9))
        times = times[keep]

        # Each shape contributes 0 outside its own support; superimpose them.
        amplitudes = torch.stack(
            [_torch_interp(times, tt, wf) for tt, wf in shapes]
        ).sum(0)

        return make_extended_trapezoid(
            channel=channel, amplitudes=amplitudes, times=times, system=system
        )

    # =========================================================================
    # At least one arbitrary gradient: rasterize all shapes and superimpose.
    # =========================================================================
    common_delay = min(g.delay for g in grads)
    total_duration = max(g.duration for g in grads)
    n = round((float(total_duration) - float(common_delay)) / system.grad_raster_time)
    grid = (torch.arange(n) + 0.5) * system.grad_raster_time

    shapes = [_control_points(g) for g in grads]
    waveform = torch.stack(
        [_torch_interp(grid, tt - common_delay, wf) for tt, wf in shapes]
    ).sum(0)

    # first/last are the summed edge values of the shapes that actually start at
    # common_delay / end at total_duration.
    first = sum(g.first for g in grads if float(g.delay) == float(common_delay))
    last = sum(g.last for g in grads if float(g.duration) == float(total_duration))

    return make_arbitrary_grad(
        channel=channel,
        waveform=waveform,
        system=system,
        max_slew=max_slew,
        max_grad=max_grad,
        delay=common_delay,
        first=first,
        last=last,
    )


# =============================================================================
# Helper functions, not exported directly
# =============================================================================


def _torch_interp(x, xp, fp):
    """torch replacement for numpys interp, returning 0 outside [xp[0], xp[-1]].
    Differentiable in x, xp and fp. The support test carries a few-ULP slack so a
    sample landing on a boundary survives float rounding (xp and x may even be
    computed in different dtypes) - important for shapes with non-zero edges."""
    x, xp, fp = torch.as_tensor(x), torch.as_tensor(xp), torch.as_tensor(fp)
    xp = xp.to(x.dtype)
    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[:-1] - m * xp[:-1]  # offset

    idx = (torch.searchsorted(xp, x, right=False) - 1).clamp(0, xp.numel() - 2)
    y = m[idx] * x + b[idx]
    tol = 8 * torch.finfo(x.dtype).eps * xp.abs().amax()  # << raster, >> float noise
    inside = (x >= xp[0] - tol) & (x <= xp[-1] + tol)
    return torch.where(inside, y, y.new_zeros(()))


def _all_traps(grads) -> TypeGuard[list[TrapGrad]]:
    return all(isinstance(g, TrapGrad) for g in grads)


def _cumsum(*args: Scalar) -> list[Scalar]:
    result = [args[0]]
    for arg in args[1:]:
        result.append(result[-1] + arg)
    return result


def _control_points(
    g: TrapGrad | ExtTrapGrad | ArbitraryGrad,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A gradient as piecewise-linear control points (absolute times, amplitudes);
    both tensors keep autograd from the event's fields. (stack promotes dtypes.)"""
    if isinstance(g, TrapGrad):
        if g.flat_time > 0:  # trapezoid
            tt = _cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time)
            wf = [0.0, g.amplitude, g.amplitude, 0.0]
        else:  # triangle
            tt = _cumsum(g.delay, g.rise_time, g.fall_time)
            wf = [0.0, g.amplitude, 0.0]
        return (
            torch.stack([torch.as_tensor(v) for v in tt]),
            torch.stack([torch.as_tensor(v) for v in wf]),
        )
    if isinstance(g, ExtTrapGrad):
        # absolute times and per-point amplitudes stored verbatim
        return torch.as_tensor(g._times), torch.as_tensor(g.waveform)
    # ArbitraryGrad: its samples define a piecewise-linear shape at absolute times.
    return torch.as_tensor(g.delay) + torch.as_tensor(g.tt), torch.as_tensor(g.waveform)
