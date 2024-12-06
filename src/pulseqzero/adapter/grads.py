from dataclasses import dataclass
from copy import copy
import numpy as np
from ..adapter import Opts, calc_duration
from ..math import ceil


def scale_grad(grad, scale):
    grad = copy(grad)
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

    ramp_up = make_extended_trapezoid(
        channel=grad.channel,
        amplitudes=np.array([0, grad.amplitude]),
        times=np.array([0, grad.rise_time])
    )
    flat_top = make_extended_trapezoid(
        channel=grad.channel,
        amplitudes=np.array([grad.amplitude, grad.amplitude]),
        times=np.array([grad.rise_time, grad.rise_time + grad.flat_time])
    )
    ramp_down = make_extended_trapezoid(
        channel=grad.channel,
        amplitudes=np.array([grad.amplitude, 0]),
        times=np.array([grad.rise_time + grad.flat_time, total_duration])
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


def make_arbitrary_grad(
    channel,
    waveform,
    delay=0,
    max_grad=None,
    max_slew=None,
    system=None,
):
    if system is None:
        system = Opts.default

    tt = (np.arange(len(waveform)) + 0.5) * system.grad_raster_time

    return FreeGrad(
        channel,
        waveform,
        delay,
        tt,
        len(waveform) * system.grad_raster_time
    )


@dataclass
class FreeGrad:
    channel: ...
    waveform: ...
    delay: ...
    tt: ...
    shape_dur: ...

    @property
    def duration(self):
        return self.delay + self.shape_dur

    @property
    def area(self):
        return 0.5 * (
            (self.tt[1:] - self.tt[:-1]) *
            (self.waveform[1:] + self.waveform[:-1])
        ).sum()


def make_extended_trapezoid(
    channel,
    amplitudes=np.zeros(1),
    convert_to_arbitrary=False,
    max_grad=None,
    max_slew=None,
    skip_check=False,
    system=None,
    times=np.zeros(1),
):
    return FreeGrad(
        channel,
        amplitudes,
        times[0],
        times - times[0],
        times[-1]
    )
