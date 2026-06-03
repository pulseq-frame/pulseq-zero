import torch
from typing import Union, cast, Literal, Optional, overload
from pypulseq import Opts
from ..events import TrapGrad, Scalar, ExtTrapGrad, Array, ArbitraryGrad
from .grad_funcs import points_to_waveform
import numpy as np


def make_trapezoid(
    channel: str,
    amplitude: Union[Scalar, None] = None,
    area: Union[Scalar, None] = None,
    delay: Scalar = 0.0,
    duration: Union[Scalar, None] = None,
    fall_time: Union[Scalar, None] = None,
    flat_area: Union[Scalar, None] = None,
    flat_time: Union[Scalar, None] = None,
    max_grad: Union[Scalar, None] = None,
    max_slew: Union[Scalar, None] = None,
    rise_time: Union[Scalar, None] = None,
    system: Union[Opts, None] = None,
) -> TrapGrad:
    if channel not in ["x", "y", "z"]:
        raise ValueError(
            f"Invalid channel. Must be one of `x`, `y` or `z`. Passed: {channel}"
        )
    if (
        flat_time is not None
        and flat_area is None
        and amplitude is None
        and (rise_time is None and fall_time is None or area is None)
    ):
        raise ValueError(
            "When `flat_time` is provided, either `flat_area`, "
            "or `amplitude`, or `rise_time` and `area` must be provided as well."
        )

    # set parameters from provided values
    if system is None:
        system = Opts.default
    if max_grad is None:
        max_grad = cast(float, system.max_grad)
    if max_slew is None:
        max_slew = cast(float, system.max_slew)
    if fall_time is None:
        fall_time = rise_time
    elif rise_time is None:
        rise_time = fall_time

    # calc_path == "area"
    if area is not None and flat_area is None and amplitude is None:
        # only duration -> fastest slews, longest flat_time
        if duration is not None and flat_time is None:
            if rise_time is None or fall_time is None:
                _, rise_time, _, fall_time = calculate_shortest_params_for_area(
                    area, max_slew, max_grad
                )
            flat_time = duration - rise_time - fall_time
            amplitude2 = area / (rise_time / 2 + fall_time / 2 + flat_time)

        # compute from flat_time
        elif flat_time is not None:
            if rise_time is None:
                raise ValueError(
                    "Must supply `rise_time` when `area` and `flat_time` is provided."
                )
            amplitude2 = area / (rise_time + flat_time)

        # no timing given -> compute shortest possible gradient
        else:
            amplitude2, rise_time, flat_time, fall_time = (
                calculate_shortest_params_for_area(area, max_slew, max_grad)
            )

    # calc_path == "flat_area"
    elif area is None and flat_area is not None and amplitude is None:
        if flat_time is None:
            raise ValueError(
                "When `flat_area` is provided, `flat_time` must be as well."
            )
        amplitude2 = flat_area / flat_time

    # calc_path == "amplitude"
    elif area is None and flat_area is None and amplitude is not None:
        if rise_time is None or fall_time is None:
            rise_time = abs(amplitude) / max_slew
            fall_time = rise_time

        amplitude2 = amplitude
        if duration is not None and flat_time is None:
            flat_time = duration - rise_time - fall_time
        elif flat_time is not None and duration is None:
            pass
        else:
            raise ValueError("Must supply area or duration.")

    elif area is None and flat_area is not None and amplitude is not None:
        raise NotImplementedError(
            "Flat Area + Amplitude input pair is not implemented yet."
        )

    else:
        raise ValueError("Must supply either 'area', 'flat_area' or 'amplitude'.")

    if rise_time is None or fall_time is None:
        rise_time = fall_time = abs(amplitude2) / max_slew

    return TrapGrad(
        channel=channel,
        amplitude=amplitude2,
        rise_time=rise_time,
        flat_time=flat_time,
        fall_time=fall_time,
        delay=delay,
    )


def calculate_shortest_params_for_area(
    area: Scalar, max_slew: Scalar, max_grad: Scalar
) -> tuple[Scalar, Scalar, Scalar, Scalar]:
    rise_time = (abs(area) / max_slew) ** 0.5

    # Calculate initial amplitude
    amplitude = area / rise_time
    effective_time = rise_time

    # Adjust for max gradient constraint
    if abs(amplitude) > max_grad:
        effective_time = abs(area) / max_grad
        amplitude = area / effective_time
        rise_time = abs(amplitude) / max_slew

    # Calculate flat and fall times
    flat_time = effective_time - rise_time
    fall_time = rise_time

    return amplitude, rise_time, flat_time, fall_time


def make_arbitrary_grad(
    channel: str,
    waveform: Array,
    first: Union[Scalar, None] = None,
    last: Union[Scalar, None] = None,
    delay: Scalar = 0.0,
    max_grad: Union[Scalar, None] = None,
    max_slew: Union[Scalar, None] = None,
    system: Union[Opts, None] = None,
    oversampling: bool = False,
) -> ArbitraryGrad:
    if system is None:
        system = Opts.default

    if channel not in ["x", "y", "z"]:
        raise ValueError(
            f"Invalid channel. Must be one of x, y or z. Passed: {channel}"
        )
    if oversampling and len(waveform) % 2 == 0:
        raise ValueError(
            "When oversampling is active, waveform must have an odd number of samples"
        )

    def extrap(a, b):
        if oversampling:
            return 2 * a - b
        else:
            return 0.5 * (3 * a - b)

    if first is None:
        first = extrap(waveform[0], waveform[1])
    if last is None:
        last = extrap(waveform[-1], waveform[-2])

    return ArbitraryGrad(
        channel=channel,
        waveform=waveform,
        delay=delay,
        first=first,
        last=last,
        oversampling=oversampling,
        _grad_raster=system.grad_raster_time,
    )


@overload
def make_extended_trapezoid(
    channel: str,
    amplitudes: Optional[Array] = ...,
    convert_to_arbitrary: Literal[False] = ...,
    max_grad: Scalar = ...,
    max_slew: Scalar = ...,
    skip_check: bool = ...,
    system: Optional[Opts] = ...,
    times: Optional[Array] = ...,
) -> ExtTrapGrad: ...
@overload
def make_extended_trapezoid(
    channel: str,
    amplitudes: Optional[Array] = ...,
    convert_to_arbitrary: Literal[True] = ...,
    max_grad: Scalar = ...,
    max_slew: Scalar = ...,
    skip_check: bool = ...,
    system: Optional[Opts] = ...,
    times: Optional[Array] = ...,
) -> ArbitraryGrad: ...
def make_extended_trapezoid(
    channel: str,
    amplitudes: Optional[Array] = None,
    convert_to_arbitrary: bool = False,
    max_grad: Scalar = 0.0,
    max_slew: Scalar = 0.0,
    skip_check: bool = False,
    system: Optional[Opts] = None,
    times: Optional[Array] = None,
) -> ArbitraryGrad | ExtTrapGrad:
    if amplitudes is None:
        amplitudes = np.zeros(1)
    if times is None:
        times = np.zeros(1)
    if system is None:
        system = Opts.default
    if max_grad <= 0:
        max_grad = cast(Scalar, system.max_grad)
    if max_slew <= 0:
        max_slew = cast(Scalar, system.max_slew)

    if channel not in ["x", "y", "z"]:
        raise ValueError(
            f"Invalid channel. Must be one of 'x', 'y' or 'z'. Passed: {channel}"
        )
    if len(times) != len(amplitudes):
        raise ValueError("Times and amplitudes must have the same length.")

    if convert_to_arbitrary:
        # Represent the extended trapezoid on the regularly sampled time grid
        # Time regridding is not differentiable
        if isinstance(times, torch.Tensor) and times.requires_grad:
            raise ValueError(
                "make_extended_trapezoid(convert_to_arbitrary=True):"
                "time regridding is not differentiable but times.requires_grad == True"
            )
        waveform = points_to_waveform(
            times=np.asarray(times),
            amplitudes=amplitudes,
            grad_raster_time=system.grad_raster_time,
        )
        return make_arbitrary_grad(
            channel=channel,
            waveform=waveform,
            system=system,
            max_slew=max_slew,
            max_grad=max_grad,
            delay=times[0],
        )
    else:
        return ExtTrapGrad(channel=channel, waveform=amplitudes, _times=times)
