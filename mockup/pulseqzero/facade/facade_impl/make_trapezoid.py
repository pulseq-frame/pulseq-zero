import torch
import numpy as np
from pypulseq import Opts


"""This code is a near identical copy of pypulseq 1.3.1post1, except that it is
compatible with torch and does not round to raster times, as well as that it
returns a class instead of a SimpleNamespace."""


class Trapezoid:
    def __init__(self, channel, delay, rise_time, flat_time, fall_time, amplitude) -> None:
        self.channel = channel
        self.delay = delay
        self.rise_time = rise_time
        self.flat_time = flat_time
        self.fall_time = fall_time
        self.amplitude = amplitude
    
    @property
    def area(self):
        return self.amplitude * (self.rise_time / 2 + self.flat_time + self.fall_time / 2)
    
    @property
    def flat_area(self):
        return self.amplitude * self.flat_time


# Need torch for differentiability, but will crash if not tensor...
def sqrt(x):
    try:
        return torch.sqrt(x)
    except:
        return np.sqrt(x)


def make_trapezoid(channel: str, amplitude: float = 0, area: float = None, delay: float = 0, duration: float = 0,
                   flat_area: float = 0, flat_time: float = -1, max_grad: float = 0, max_slew: float = 0,
                   rise_time: float = 0, system: Opts = Opts()) -> Trapezoid:
    if channel not in ['x', 'y', 'z']:
        raise ValueError(f"Invalid channel. Must be one of `x`, `y` or `z`. Passed: {channel}")

    if max_grad <= 0:
        max_grad = system.max_grad

    if max_slew <= 0:
        max_slew = system.max_slew

    if rise_time <= 0:
        rise_time = system.rise_time

    if area is None and flat_area == 0 and amplitude == 0:
        raise ValueError("Must supply either 'area', 'flat_area' or 'amplitude'.")

    if flat_time != -1:
        if amplitude != 0:
            amplitude2 = amplitude
        else:
            amplitude2 = flat_area / flat_time

        if rise_time == 0:
            rise_time = abs(amplitude2) / max_slew
        fall_time, flat_time = rise_time, flat_time
    elif duration > 0:
        amplitude2 = amplitude
        if amplitude == 0:
            if rise_time == 0:
                dC = 1 / abs(2 * max_slew) + 1 / abs(2 * max_slew)
                possible = duration ** 2 > 4 * abs(area) * dC
                amplitude2 = (duration - sqrt(duration *2 - 4 * abs(area) * dC)) / (2 * dC)
            else:
                amplitude2 = area / (duration - rise_time)
                possible = duration > 2 * rise_time and abs(amplitude2) < max_grad

            if not possible:
                raise ValueError('Requested area is too large for this gradient')

        if rise_time == 0:
            rise_time = abs(amplitude2) / max_slew
            if rise_time == 0:
                rise_time = system.grad_raster_time

        fall_time = rise_time
        flat_time = duration - rise_time - fall_time

        if amplitude == 0:
            amplitude2 = area / (rise_time / 2 + fall_time / 2 + flat_time)
    else:
        if area == 0:
            raise ValueError('Must supply a duration.')
        else:
            rise_time = sqrt(abs(area) / max_slew)
            amplitude2 = np.divide(area, rise_time)  # To handle nan
            t_eff = rise_time

            if abs(amplitude2) > max_grad:
                t_eff = abs(area) / max_grad
                amplitude2 = area / t_eff
                rise_time = abs(amplitude2) / max_slew

            flat_time = t_eff - rise_time
            fall_time = rise_time

    if abs(amplitude2) > max_grad:
        raise ValueError("Amplitude violation.")
    
    return Trapezoid(channel, delay, rise_time, flat_time, fall_time, amplitude2)
