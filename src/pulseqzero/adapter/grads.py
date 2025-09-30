from dataclasses import dataclass
from copy import copy, deepcopy
import torch
from ..adapter import Opts, calc_duration
from ..math import ceil


def scale_grad(grad, scale):
    grad = deepcopy(grad)
    if isinstance(grad, TrapGrad):
        grad.amplitude *= scale
    if isinstance(grad, FreeGrad):
        grad.waveform *= scale
    return grad


def split_gradient(grad, system):
    assert isinstance(grad, TrapGrad)
    if system is None:
        system = Opts.default
    total_duration = calc_duration(grad)

    def join(*args):
        return torch.stack([torch.as_tensor(x) for x in args])

    ramp_up = make_extended_trapezoid(
        channel=grad.channel,
        amplitudes=join(0, grad.amplitude),
        times=join(0, grad.rise_time)
    )
    flat_top = make_extended_trapezoid(
        channel=grad.channel,
        amplitudes=join(grad.amplitude, grad.amplitude),
        times=join(grad.rise_time, grad.rise_time + grad.flat_time)
    )
    ramp_down = make_extended_trapezoid(
        channel=grad.channel,
        amplitudes=join(grad.amplitude, 0),
        times=join(grad.rise_time + grad.flat_time, total_duration)
    )

    return ramp_up, flat_top, ramp_down


def make_trapezoid(
    channel,
    amplitude=None,
    area=None,
    delay=0,
    duration=None,
    fall_time=None,
    flat_area=None,
    flat_time=None,
    max_grad=None,
    max_slew=None,
    rise_time=None,
    system=None,
):
    if system is None:
        system = Opts.default
    if max_grad is None:
        max_grad = system.max_grad
    if max_slew is None:
        max_slew = system.max_slew

    # new_amp is only calculated to set rise_time below, the actual
    # amplitude is then calculated from tmp_amp and the timing

    # TODO: This function should really be split into multiple with the different argument combination options

    if flat_time is not None:
        if amplitude is not None:
            new_amp = amplitude
        elif area is not None:
            assert rise_time is not None
            if fall_time is None:
                fall_time = rise_time
            new_amp = area / (rise_time / 2 + flat_time + fall_time / 2)
        else:
            assert flat_area is not None
            new_amp = flat_area / flat_time

        if rise_time is None:
            rise_time = ceil(abs(new_amp) / max_slew / system.grad_raster_time) * system.grad_raster_time
            if rise_time == 0:
                rise_time = system.grad_raster_time
        if fall_time is None:
            fall_time = rise_time

    elif duration is not None:
        if amplitude is None:
            assert area is not None

            if rise_time is None:
                _, rise_time, flat_time, fall_time = calc_params_for_area(
                    area, max_slew, max_grad, system.grad_raster_time
                )
                assert duration >= rise_time + flat_time + fall_time

                dC = 1 / abs(2 * max_slew)
                new_amp = (
                    duration - (duration**2 - 4 * abs(area) * dC)
                ) / (2 * dC)
            else:
                if fall_time is None:
                    fall_time = rise_time
                new_amp = area / (duration - rise_time / 2 - fall_time / 2)
        else:
            new_amp = amplitude

        if rise_time is None:
            rise_time = ceil(abs(new_amp) / max_slew / system.grad_raster_time) * system.grad_raster_time
            if rise_time == 0:
                rise_time = system.grad_raster_time
        if fall_time is None:
            fall_time = rise_time
        flat_time = duration - rise_time - fall_time

        if amplitude is None:
            new_amp = area / (rise_time / 2 + flat_time + fall_time / 2)

    else:
        assert area is not None
        new_amp, rise_time, flat_time, fall_time = calc_params_for_area(
            area, max_slew, max_grad, system.grad_raster_time
        )

    return TrapGrad(
        channel,
        new_amp,
        rise_time,
        flat_time,
        fall_time,
        delay
    )


def calc_params_for_area(area, max_slew, max_grad, grad_raster_time):
    rise_time = ceil((abs(area) / max_slew)**0.5 / grad_raster_time) * grad_raster_time
    amplitude = area / rise_time
    t_eff = rise_time

    if abs(amplitude) > max_grad:
        t_eff = ceil(abs(area) / max_grad / grad_raster_time) * grad_raster_time
        amplitude = area / t_eff
        rise_time = ceil(abs(amplitude) / max_slew / grad_raster_time) * grad_raster_time

        if rise_time == 0:
            rise_time = grad_raster_time

    flat_time = t_eff - rise_time
    fall_time = rise_time

    return amplitude, rise_time, flat_time, fall_time


@dataclass
class TrapGrad:
    channel: ...
    amplitude: ...
    rise_time: ...
    flat_time: ...
    fall_time: ...
    delay: ...

    @property
    def area(self):
        return self.amplitude * (self.rise_time / 2 + self.flat_time + self.fall_time / 2)

    @property
    def flat_area(self):
        return self.amplitude * self.flat_time

    @property
    def duration(self):
        return self.delay + self.rise_time + self.flat_time + self.fall_time
    
    @property
    def first(self):
        return 0.0
    
    @property
    def last(self):
        return 0.0


def make_arbitrary_grad(
    channel,
    waveform,
    delay=0,
    max_grad=None,
    max_slew=None,
    system=None,
    first=None,
    last=None
):
    if system is None:
        system = Opts.default

    tt = (torch.arange(len(waveform)) + 0.5) * system.grad_raster_time

    if first is None:
        first = (3 * waveform[0] - waveform[1]) * 0.5  # linear extrapolation

    if last is None:
        last = (3 * waveform[-1] - waveform[-2]) * 0.5  # linear extrapolation


    return FreeGrad(
        channel,
        waveform,
        delay,
        tt,
        len(waveform) * system.grad_raster_time,
        first, 
        last
    )


@dataclass
class FreeGrad:
    channel: ...
    waveform: ...
    delay: ...
    tt: ...
    shape_dur: ...
    first_waveform: ...
    last_waveform: ...

    @property
    def duration(self):
        return self.delay + self.shape_dur

    @property
    def area(self):
        return 0.5 * (
            (self.tt[1:] - self.tt[:-1]) *
            (self.waveform[1:] + self.waveform[:-1])
        ).sum()
    
    @property
    def first(self):
        return self.first_waveform 
    
    @property
    def last(self):
        return self.last_waveform 


def  make_extended_trapezoid(
    channel,
    amplitudes=torch.zeros(1),
    convert_to_arbitrary=False,
    max_grad=None,
    max_slew=None,
    skip_check=False,
    system=None,
    times=torch.zeros(1),
):
    return FreeGrad(
        channel,
        amplitudes,
        times[0],
        times - times[0],
        times[-1], 
        amplitudes[0],
        amplitudes[-1]
    )


def add_gradients(
        grads,
        max_grad=None,
        max_slew=None,
        system=None,
):
    if len(grads) == 0:
        raise ValueError("No gradients specified")
    if len(grads) == 1:
        return deepcopy(grads[0])

    channel = grads[0].channel
    if any(g.channel != channel for g in grads):
        raise ValueError("Cannot add gradients on different channels")
    
    if all((isinstance(g, TrapGrad) and
            g.rise_time == grads[0].rise_time and
            g.flat_time == grads[0].flat_time and
            g.fall_time == grads[0].fall_time and
            g.delay == grads[0].delay) for g in grads):
        return make_trapezoid(
            channel=channel,
            amplitude=sum(g.amplitude for g in grads),
            rise_time=grads[0].rise_time,
            flat_time=grads[0].flat_time,
            fall_time=grads[0].fall_time,
            delay=grads[0].delay,
            max_grad=max_grad,
            max_slew=max_slew,
            system=system,
        )

    if system is None:
        system = Opts.default
    if max_grad is None:
        max_grad = system.max_grad
    if max_slew is None:
        max_slew = system.max_slew
    

    # !!! Non-differentiable pulseq 1.4.2 code !!!
    # - start with a diff check to raise an error
    def requires_grad(x):
        return torch.as_tensor(x).requires_grad
    
    def check_req_grad(grad):
        if isinstance(grad, TrapGrad):
            for attr in ["amplitude", "delay", "rise_time", "flat_time", "fall_time"]:
                if requires_grad(getattr(grad, attr)):
                    return attr
        elif isinstance(grad, FreeGrad):
            for attr in ["waveform", "delay", "tt"]:
                if requires_grad(getattr(grad, attr)):
                    return attr
        else:
            raise TypeError(f"Expected TrapGrad or FreeGrad, got {type(grad)}")

    for grad in grads:
        attr = check_req_grad(grad)
        if attr is not None:
            raise ValueError(f"add_gradients() is not yet differentiable, but {attr} of {grad} has requires_grad=True. File an issue if you need this functionality.")
            
    def points_to_waveform(amplitudes, grad_raster_time, times):
        amplitudes = torch.as_tensor(amplitudes, dtype=torch.float64)
        times = torch.as_tensor(times, dtype=torch.float64)

        if amplitudes.numel() == 0:
            return torch.tensor([0.0], dtype=torch.float64)

        start_val = round((torch.min(times) / grad_raster_time).item())
        stop_val = round((torch.max(times) / grad_raster_time).item())
        grd = torch.arange(start_val, stop_val, dtype=torch.float64) * grad_raster_time

        # PyTorch equivalent of np.interp
        x_vals = grd + grad_raster_time / 2
        waveform = torch.zeros_like(x_vals)
        
        for i, x in enumerate(x_vals):
            # Find the interpolation indices
            idx = torch.searchsorted(times, x, right=False)
            if len(times) == 0:
                waveform[i] = 0.0
            elif idx == 0:
                if x <= times[0]:
                    waveform[i] = amplitudes[0]
                else:
                    waveform[i] = amplitudes[0]
            elif idx >= len(times):
                waveform[i] = amplitudes[-1]
            else:
                # Linear interpolation
                t0, t1 = times[idx-1], times[idx]
                a0, a1 = amplitudes[idx-1], amplitudes[idx]
                if t1 != t0:
                    alpha = (x - t0) / (t1 - t0)
                    waveform[i] = a0 + alpha * (a1 - a0)
                else:
                    waveform[i] = a0

        return waveform

    # - copy of pulseq code
    import numpy as np
    eps = 1e-9
    def cumsum(args):
        return torch.cumsum(args)

    # Find out the general delay of all gradients and other statistics
    delays, firsts, lasts, durs, is_trap, is_arb = [], [], [], [], [], []
    for ii in range(len(grads)):
        if grads[ii].channel != channel:
            raise ValueError("Cannot add gradients on different channels.")

        delays.append(grads[ii].delay)
        firsts.append(grads[ii].first)
        lasts.append(grads[ii].last)
        durs.append(calc_duration(grads[ii]))
        is_trap.append(isinstance(grads[ii], TrapGrad))
        if is_trap[-1]:
            is_arb.append(False)
        else:
            tt_rast = grads[ii].tt / system.grad_raster_time - 0.5
            is_arb.append(torch.all(torch.abs(tt_rast - torch.arange(len(tt_rast)))) < eps)
    is_arb = torch.as_tensor(is_arb)
    is_trap = torch.as_tensor(is_trap)

    # Check if we only have arbitrary grads on irregular time samplings, optionally mixed with trapezoids
    if torch.all(torch.logical_or(is_trap, torch.logical_not(is_arb))):
        # Keep shapes still rather simple
        times = []
        for ii in range(len(grads)):
            g = grads[ii]
            if isinstance(g, TrapGrad):
                times.extend(
                    cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time)
                )
            else:
                times.extend(g.delay + g.tt)

        times = torch.unique(torch.tensor(times))
        dt = times[1:] - times[:-1]
        ieps = torch.nonzero(dt < eps, as_tuple=False).flatten()
        if torch.any(ieps):
            dtx = torch.tensor([times[0], *dt])
            dtx[ieps] = (
                dtx[ieps] + dtx[ieps + 1]
            )  # Assumes that no more than two too similar values can occur
            dtx = torch.delete(dtx, ieps + 1)
            times = torch.cumsum(dtx)

        amplitudes = torch.zeros_like(times)
        for ii in range(len(grads)):
            g = grads[ii]
            if isinstance(g, TrapGrad):
                if g.flat_time > 0:  # Trapezoid or triangle
                    tt = cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time)
                    waveform = torch.tensor([0, g.amplitude, g.amplitude, 0], dtype=torch.float64)
                else:
                    tt = cumsum(g.delay, g.rise_time, g.fall_time)
                    waveform = torch.tensor([0, g.amplitude, 0], dtype=torch.float64)
            else:
                tt = g.delay + g.tt
                waveform = g.waveform

            # Fix rounding for the first and last time points
            i_min = torch.argmin(torch.abs(tt[0] - times))
            t_min = torch.abs(tt[0] - times)[i_min]
            if t_min < eps:
                tt[0] = times[i_min]
            i_min = torch.argmin(torch.abs(tt[-1] - times))
            t_min = torch.abs(tt[-1] - times)[i_min]
            if t_min < eps:
                tt[-1] = times[i_min]

            if abs(waveform[0]) > eps and tt[0] > eps:
                tt[0] += eps

            # PyTorch equivalent of np.interp with left=0, right=0
            interp_values = torch.zeros_like(times)
            for i, x in enumerate(times):
                idx = torch.searchsorted(tt, x)
                if idx == 0:
                    if x <= tt[0]:
                        interp_values[i] = 0.0  # left=0
                    else:
                        interp_values[i] = waveform[0]
                elif idx >= len(tt):
                    if x >= tt[-1]:
                        interp_values[i] = 0.0  # right=0
                    else:
                        interp_values[i] = waveform[-1]
                else:
                    # Linear interpolation
                    t0, t1 = tt[idx-1], tt[idx]
                    w0, w1 = waveform[idx-1], waveform[idx]
                    alpha = (x - t0) / (t1 - t0)
                    interp_values[i] = w0 + alpha * (w1 - w0)
            
            amplitudes += interp_values

        grad = make_extended_trapezoid(
            channel=channel, amplitudes=amplitudes, times=times, system=system
        )
        return grad
    
    # Convert to torch tensors for indexing later on
    firsts, lasts = torch.tensor(firsts, dtype=torch.float64), torch.tensor(lasts, dtype=torch.float64)
    common_delay = min(delays)
    durs = torch.tensor(durs, dtype=torch.float64)

    # Convert everything to a regularly-sampled waveform
    waveforms = dict()
    max_length = 0
    for ii in range(len(grads)):
        g = grads[ii]
        if isinstance(g, FreeGrad):
            if is_arb[ii]:
                waveforms[ii] = g.waveform
            else:
                waveforms[ii] = points_to_waveform(
                    amplitudes=g.waveform,
                    times=g.tt,
                    grad_raster_time=system.grad_raster_time,
                )
        elif isinstance(g, TrapGrad):
            if g.flat_time > 0:  # Triangle or trapezoid
                times = torch.tensor(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.flat_time,
                        g.delay
                        - common_delay
                        + g.rise_time
                        + g.flat_time
                        + g.fall_time,
                    ], dtype=torch.float64
                )
                amplitudes = torch.tensor([0, g.amplitude, g.amplitude, 0], dtype=torch.float64)
            else:
                times = torch.tensor(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.fall_time,
                    ], dtype=torch.float64
                )
                amplitudes = torch.tensor([0, g.amplitude, 0], dtype=torch.float64)
            waveforms[ii] = points_to_waveform(
                amplitudes=amplitudes,
                times=times,
                grad_raster_time=system.grad_raster_time,
            )
        else:
            raise ValueError("Unknown gradient type")

        if g.delay - common_delay > 0:
            # Create delay samples using torch.arange
            t_delay = torch.arange(0, g.delay - common_delay, step=system.grad_raster_time, dtype=torch.float64)
            waveforms[ii] = torch.cat([torch.zeros_like(t_delay), waveforms[ii]])

        num_points = len(waveforms[ii])
        max_length = max(num_points, max_length)

    w = torch.zeros(max_length, dtype=torch.float64)
    for ii in range(len(grads)):
        wt = torch.zeros(max_length, dtype=torch.float64)
        waveform_len = len(waveforms[ii])
        wt[:waveform_len] = waveforms[ii]
        w += wt

    grad = make_arbitrary_grad(
        channel=channel,
        waveform=w,
        system=system,
        max_slew=max_slew,
        max_grad=max_grad,
        delay=common_delay,
    )
    # Fix the first and the last values
    # First is defined by the sum of firsts with the minimal delay (common_delay)
    # Last is defined by the sum of lasts with the maximum duration (total_duration == durs.max())
    delays_tensor = torch.tensor(delays, dtype=torch.float64)
    grad.first = torch.sum(firsts[delays_tensor == common_delay])
    grad.last = torch.sum(lasts[durs == torch.max(durs)])

    return grad
