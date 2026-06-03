import torch
from typing import cast, Literal, Optional, overload
from pypulseq import Opts
from ..events import TrapGrad, Scalar, ExtTrapGrad, Array, ArbitraryGrad
from .grad_funcs import points_to_waveform
import numpy as np


def make_trapezoid(
    channel: str,
    amplitude: Optional[Scalar] = None,
    area: Optional[Scalar] = None,
    delay: Scalar = 0.0,
    duration: Optional[Scalar] = None,
    fall_time: Optional[Scalar] = None,
    flat_area: Optional[Scalar] = None,
    flat_time: Optional[Scalar] = None,
    max_grad: Optional[Scalar] = None,
    max_slew: Optional[Scalar] = None,
    rise_time: Optional[Scalar] = None,
    system: Optional[Opts] = None,
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
    first: Optional[Scalar] = None,
    last: Optional[Scalar] = None,
    delay: Scalar = 0.0,
    max_grad: Optional[Scalar] = None,
    max_slew: Optional[Scalar] = None,
    system: Optional[Opts] = None,
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


@overload
def make_extended_trapezoid_area(
    area: Scalar,
    channel: str,
    grad_start: Scalar,
    grad_end: Scalar,
    convert_to_arbitrary: Literal[False] = ...,
    system: Optional[Opts] = ...,
    duration: Optional[Scalar] = ...,
    max_grad: Optional[Scalar] = ...,
    max_slew: Optional[Scalar] = ...,
) -> tuple[ExtTrapGrad, Array, Array]: ...
@overload
def make_extended_trapezoid_area(
    area: Scalar,
    channel: str,
    grad_start: Scalar,
    grad_end: Scalar,
    convert_to_arbitrary: Literal[True] = ...,
    system: Optional[Opts] = ...,
    duration: Optional[Scalar] = ...,
    max_grad: Optional[Scalar] = ...,
    max_slew: Optional[Scalar] = ...,
) -> tuple[ArbitraryGrad, Array, Array]: ...
def make_extended_trapezoid_area(
    area: Scalar,
    channel: str,
    grad_start: Scalar,
    grad_end: Scalar,
    convert_to_arbitrary: bool = False,
    system: Optional[Opts] = None,
    duration: Optional[Scalar] = None,
    max_grad: Optional[Scalar] = None,
    max_slew: Optional[Scalar] = None,
) -> tuple[ArbitraryGrad | ExtTrapGrad, Array, Array]:
    from .grad_funcs import _cumsum

    if system is None:
        system = Opts.default
    if max_grad is None:
        max_grad = system.max_grad
    if max_slew is None:
        max_slew = system.max_slew

    if channel not in ["x", "y", "z"]:
        raise ValueError(
            f"Invalid channel. Must be one of `x`, `y` or `z`. Passed: {channel}"
        )
    if duration is not None and duration <= 0:
        raise ValueError("Duration must be a positive number.")

    raster_time = system.grad_raster_time

    def _to_raster(time: float) -> float:
        return np.ceil(time / raster_time) * raster_time

    def _calc_ramp_time(grad_1: float, grad_2: float) -> float:
        return _to_raster(abs(grad_1 - grad_2) / max_slew)

    def _find_solution(duration: int) -> Union[None, Tuple[int, int, int, float]]:
        """Find extended trapezoid gradient waveform for given duration.

        The function performs a grid search over all possible ramp-up, ramp-down and flat times
        for the given duration and returns the solution with the lowest slew rate.

        Parameters
        ----------
        duration
            duration of the gradient in integer multiples of raster_time

        Returns
        -------
            Tuple of ramp-up time, flat time, ramp-down time, gradient amplitude or None if no solution was found
        """
        # Determine timings to check for possible solutions
        ramp_up_times = []
        ramp_down_times = []

        # First, consider solutions that use maximum slew rate:
        # Analytically calculate calculate the point where:
        #   grad_start + ramp_up_time * max_slew == grad_end + ramp_down_time * max_slew
        ramp_up_time = (duration * max_slew * raster_time - grad_start + grad_end) / (
            2 * max_slew * raster_time
        )
        ramp_up_time = round(ramp_up_time)

        # Check if gradient amplitude exceeds max_grad, if so, adjust ramp
        # times for a trapezoidal gradient with maximum slew rate.
        if grad_start + ramp_up_time * max_slew * raster_time > max_grad + eps:
            ramp_up_time = round(_calc_ramp_time(grad_start, max_grad) / raster_time)
            ramp_down_time = round(_calc_ramp_time(grad_end, max_grad) / raster_time)
        else:
            ramp_down_time = duration - ramp_up_time

        # Add possible solution if timing is valid
        if (
            ramp_up_time > 0
            and ramp_down_time > 0
            and ramp_up_time + ramp_down_time <= duration
        ):
            ramp_up_times.append(ramp_up_time)
            ramp_down_times.append(ramp_down_time)

        # Analytically calculate calculate the point where:
        #   grad_start - ramp_up_time * max_slew == grad_end - ramp_down_time * max_slew
        ramp_up_time = (duration * max_slew * raster_time + grad_start - grad_end) / (
            2 * max_slew * raster_time
        )
        ramp_up_time = round(ramp_up_time)

        # Check if gradient amplitude exceeds -max_grad, if so, adjust ramp
        # times for a trapezoidal gradient with maximum slew rate.
        if grad_start - ramp_up_time * max_slew * raster_time < -max_grad - eps:
            ramp_up_time = round(_calc_ramp_time(grad_start, -max_grad) / raster_time)
            ramp_down_time = round(_calc_ramp_time(grad_end, -max_grad) / raster_time)
        else:
            ramp_down_time = duration - ramp_up_time

        # Add possible solution if timing is valid
        if (
            ramp_up_time > 0
            and ramp_down_time > 0
            and ramp_up_time + ramp_down_time <= duration
        ):
            ramp_up_times.append(ramp_up_time)
            ramp_down_times.append(ramp_down_time)

        # Second, try any solution with flat_time == 0
        # This appears to be necessary for many cases, but going through all
        # timings here is probably too conservative still.
        for ramp_up_time in range(1, duration):
            ramp_up_times.append(ramp_up_time)
            ramp_down_times.append(duration - ramp_up_time)

        time_ramp_up = np.array(ramp_up_times)
        time_ramp_down = np.array(ramp_down_times)

        # Calculate corresponding flat times
        flat_time = duration - time_ramp_up - time_ramp_down

        # Filter search space for valid timings (flat time >= 0)
        valid_indices = flat_time >= 0
        time_ramp_up = time_ramp_up[valid_indices]
        time_ramp_down = time_ramp_down[valid_indices]
        flat_time = flat_time[valid_indices]

        # Calculate gradient strength for given timing using analytical solution
        grad_amp = -(
            time_ramp_up * raster_time * grad_start
            + time_ramp_down * raster_time * grad_end
            - 2 * area
        ) / ((time_ramp_up + 2 * flat_time + time_ramp_down) * raster_time)

        # Calculate slew rates for given timings
        slew_rate1 = abs(grad_start - grad_amp) / (time_ramp_up * raster_time)
        slew_rate2 = abs(grad_end - grad_amp) / (time_ramp_down * raster_time)

        # Filter solutions that satisfy max_grad and max_slew constraints
        valid_indices = (
            (abs(grad_amp) <= max_grad + eps)
            & (slew_rate1 <= max_slew + eps)
            & (slew_rate2 <= max_slew + eps)
        )
        solutions = np.flatnonzero(valid_indices)

        # Check if any valid solutions were found
        if solutions.shape[0] == 0:
            return None

        # Find solution with lowest slew rate and return it
        ind = np.argmin(slew_rate1[valid_indices] + slew_rate2[valid_indices])
        ind = solutions[ind]
        return (
            int(time_ramp_up[ind]),
            int(flat_time[ind]),
            int(time_ramp_down[ind]),
            float(grad_amp[ind]),
        )

    if duration is None:  # duration was not given
        # Perform a linear search
        # This is necessary because there can exist a dead space where solutions
        # do not exist for some durations longer than the optimal duration. The
        # binary search below fails to find the optimum in those cases.
        # TODO: Check if range is sufficient, try to calculate the dead space.
        min_duration = max(
            round(_calc_ramp_time(grad_end, grad_start) / raster_time), 2
        )

        # Calculate duration needed to ramp down gradient to zero.
        # From this point onwards, solutions can always be found by extending
        # the duration and doing a binary search.
        max_duration = max(
            round(_calc_ramp_time(0, grad_start) / raster_time),
            round(_calc_ramp_time(0, grad_end) / raster_time),
            min_duration,
        )

        # Linear search
        solution = None
        for current_duration in range(min_duration, max_duration + 1):
            solution = _find_solution(current_duration)
            if solution is not None:
                break

        # Perform a binary search for duration > max_duration if no solution was found
        if solution is None:
            # First, find the upper limit on duration where a solution exists by
            # exponentially expanding the duration.
            while solution is None:
                max_duration *= 2
                solution = _find_solution(max_duration)

            def binary_search(fun, lower_limit, upper_limit):
                if lower_limit == upper_limit - 1:
                    return fun(upper_limit)

                test_value = (upper_limit + lower_limit) // 2

                if fun(test_value):
                    return binary_search(fun, lower_limit, test_value)
                else:
                    return binary_search(fun, test_value, upper_limit)

            solution = binary_search(_find_solution, max_duration // 2, max_duration)

    else:  # duration was given, so calculate solution for this duration
        duration_raster = max(round(_to_raster(duration) / raster_time), 2)
        solution = _find_solution(duration_raster)

        if solution is None:
            raise ValueError(
                f"Could not find a solution for area={area} and duration={duration}."
            )

    # Get timing and gradient amplitude from solution
    time_ramp_up = solution[0] * raster_time
    flat_time = solution[1] * raster_time
    time_ramp_down = solution[2] * raster_time
    grad_amp = solution[3]

    # Create extended trapezoid
    if flat_time > 0:
        times = _cumsum(0, time_ramp_up, flat_time, time_ramp_down)
        amplitudes = np.array([grad_start, grad_amp, grad_amp, grad_end])
    else:
        times = _cumsum(0, time_ramp_up, time_ramp_down)
        amplitudes = np.array([grad_start, grad_amp, grad_end])

    grad = make_extended_trapezoid(
        channel=channel,
        amplitudes=amplitudes,
        convert_to_arbitrary=convert_to_arbitrary,
        system=system,
        times=times,
    )

    if not abs(grad.area - area) < eps:
        raise ValueError(f"Could not find a solution for area={area}.")

    return grad, grad.tt, grad.waveform
