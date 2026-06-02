from copy import copy, deepcopy
from typing import Optional, TypeVar, cast, TypeGuard
from pypulseq import Opts
from ..events import TrapGrad, ExtTrapGrad, ArbitraryGrad, Array, Scalar
from .make_grad import make_trapezoid, make_arbitrary_grad, make_extended_trapezoid
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


def _all_traps(grads) -> TypeGuard[list[TrapGrad]]:
    return all(isinstance(g, TrapGrad) for g in grads)


def cumsum(*args: Scalar) -> list[Scalar]:
    result = [args[0]]
    for arg in args[1:]:
        result.append(result[-1] + arg)
    return result


def add_gradients(
    grads: list[TrapGrad | ExtTrapGrad | ArbitraryGrad],
    max_grad: Scalar = 0,
    max_slew: Scalar = 0,
    system: Optional[Opts] = None,
) -> TrapGrad | ExtTrapGrad | ArbitraryGrad:
    if len(grads) == 0:
        raise ValueError("No gradients specified")
    if len(grads) == 1:
        return deepcopy(grads[0])

    # all gradients must have the same channel
    channel = grads[0].channel
    if channel not in ["x", "y", "z"]:
        raise ValueError(
            f"Invalid channel. Must be one of `x`, `y` or `z`. Passed: {channel}"
        )
    if not all(g.channel == channel for g in grads):
        raise ValueError("Cannot add gradients on different channels.")

    # set defaults
    if system is None:
        system = Opts.default
    if max_grad <= 0:
        max_grad = cast(float, system.max_grad)
    if max_slew <= 0:
        max_slew = cast(float, system.max_slew)

    # Check if we have a set of traps with the same timing
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

    # Find out the general delay of all gradients and other statistics
    delays = [g.delay for g in grads]
    durs = [g.duration for g in grads]
    is_trap = [isinstance(g, TrapGrad) for g in grads]
    is_etrap = [isinstance(g, ExtTrapGrad) for g in grads]
    is_arb = [isinstance(g, ArbitraryGrad) for g in grads]
    is_osa = [isinstance(g, ArbitraryGrad) and g.oversampling for g in grads]
    firsts = [g.first for g in grads]
    lasts = [g.last for g in grads]

    # Check if we only have arbitrary grads on irregular time samplings, optionally mixed with trapezoids
    if not any(is_arb):
        # Keep shapes still rather simple
        times = []
        for g in grads:
            if isinstance(g, TrapGrad):
                times.extend(cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time))
            else:
                times.extend(g.delay + g.tt)

        times = np.unique(times)
        dt = times[1:] - times[:-1]
        ieps = np.flatnonzero(dt < eps)
        if np.any(ieps):
            dtx = np.array([times[0], *dt])
            dtx[ieps] = (
                dtx[ieps] + dtx[ieps + 1]
            )  # Assumes that no more than two too similar values can occur
            dtx = np.delete(dtx, ieps + 1)
            times = np.cumsum(dtx)

        amplitudes = np.zeros_like(times)
        for g in grads:
            if isinstance(g, TrapGrad):
                if g.flat_time > 0:  # Trapezoid or triangle
                    tt = list(cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time))
                    waveform = [0, g.amplitude, g.amplitude, 0]
                else:
                    tt = list(cumsum(g.delay, g.rise_time, g.fall_time))
                    waveform = [0, g.amplitude, 0]
            else:
                tt = g.delay + g.tt
                waveform = g.waveform

            # Fix rounding for the first and last time points
            i_min = np.argmin(np.abs(tt[0] - times))
            t_min = (np.abs(tt[0] - times))[i_min]
            if t_min < eps:
                tt[0] = times[i_min]
            i_min = np.argmin(np.abs(tt[-1] - times))
            t_min = (np.abs(tt[-1] - times))[i_min]
            if t_min < eps:
                tt[-1] = times[i_min]

            if abs(waveform[0]) > eps and tt[0] > eps:
                tt[0] += eps

            amplitudes += _torch_interp(xp=tt, fp=waveform, x=times)

        grad = make_extended_trapezoid(
            channel=channel, amplitudes=amplitudes, times=times, system=system
        )

        return grad

    # Convert to numpy.ndarray for fancy-indexing later on
    firsts, lasts = np.array(firsts), np.array(lasts)
    common_delay = np.min(delays)
    total_duration = np.max(durs)
    durs = np.array(durs)

    # Convert everything to a regularly-sampled waveform
    waveforms = {}
    max_length = 0

    if any(is_osa):
        target_raster = 0.5 * system.grad_raster_time
    else:
        target_raster = system.grad_raster_time

    for ii in range(len(grads)):
        g = grads[ii]
        if not isinstance(g, TrapGrad):
            if is_arb[ii] or is_osa[ii]:
                if (
                    np.any(is_osa) and is_arb[ii]
                ):  # Porting MATLAB here, maybe a bit ugly
                    # Interpolate missing samples
                    idx = np.arange(0, len(g.waveform) - 0.5 + eps, 0.5)
                    wf = g.waveform
                    interp_waveform = 0.5 * (
                        wf[np.floor(idx).astype(int)] + wf[np.ceil(idx).astype(int)]
                    )
                    waveforms[ii] = interp_waveform
                else:
                    waveforms[ii] = g.waveform
            else:
                waveforms[ii] = points_to_waveform(
                    amplitudes=g.waveform,
                    times=g.tt,
                    grad_raster_time=target_raster,
                )
        elif isinstance(g, TrapGrad):
            if g.flat_time > 0:  # Triangle or trapezoid
                times = np.array(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.flat_time,
                        g.delay
                        - common_delay
                        + g.rise_time
                        + g.flat_time
                        + g.fall_time,
                    ]
                )
                amplitudes = np.array([0, g.amplitude, g.amplitude, 0])
            else:
                times = np.array(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.fall_time,
                    ]
                )
                amplitudes = np.array([0, g.amplitude, 0])
            waveforms[ii] = points_to_waveform(
                amplitudes=amplitudes,
                times=times,
                grad_raster_time=target_raster,
            )

        if g.delay - common_delay > 0:
            # Stop for numpy.arange is not g.delay - common_delay - system.grad_raster_time like in Matlab
            # so as to include the endpoint
            waveforms[ii] = np.concatenate(
                (
                    np.zeros(round((g.delay - common_delay) / system.grad_raster_time)),
                    waveforms[ii],
                )
            )

        num_points = len(waveforms[ii])
        max_length = max(num_points, max_length)

    w = np.zeros(max_length)
    for ii in range(len(grads)):
        wt = np.zeros(max_length)
        wt[0 : len(waveforms[ii])] = waveforms[ii]
        w += wt

    grad = make_arbitrary_grad(
        channel=channel,
        waveform=w,
        system=system,
        max_slew=max_slew,
        max_grad=max_grad,
        delay=common_delay,
        oversampling=any(is_osa),
        first=np.sum(firsts[delays == common_delay]),
        last=np.sum(lasts[durs == total_duration]),
    )
    # Fix the first and the last values
    # First is defined by the sum of firsts with the minimal delay (common_delay)
    # Last is defined by the sum of lasts with the maximum duration (total_duration == durs.max())
    grad.first = np.sum(firsts[np.array(delays) == common_delay])
    grad.last = np.sum(lasts[durs == durs.max()])

    return grad
