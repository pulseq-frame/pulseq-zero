import pypulseq
from warnings import warn
from typing import Optional, cast
from copy import deepcopy
import numpy as np
import torch
from ..events import Scalar, RfPulse, Event, Array, SoftDelay
from pypulseq import Opts


def calc_duration(*args: Event) -> Scalar:
    return max(ev.duration for ev in args if ev is not None)


def calc_rf_bandwidth(
    rf: RfPulse, *args, **kwargs
) -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, float]:
    warn("calc_rf_bandwidth is not differentiable.")
    return pypulseq.calc_rf_bandwidth(rf.to_pulseq(Opts.default), *args, **kwargs)


def calc_rf_center(rf: RfPulse) -> tuple[Scalar, int]:
    warn(
        "pulseq-zeros calc_rf_center does not compute the center shape index (returns 0). "
        "The returned time point is computed on pulse construction - it is encouraged to use rf.center directly. "
        "This behaviour does not affect forwarded pypulseq sequence operations which use the pypulseq implementation."
    )
    return rf.center, 0


def traj_to_grad(k: Array, raster_time: Optional[float] = None) -> tuple[Array, Array]:
    if raster_time is None:
        raster_time = Opts.default.grad_raster_time

    # Compute finite difference for gradients in Hz/m
    g = (k[..., 1:] - k[..., :-1]) / raster_time
    # Compute the slew rate
    sr0 = (g[..., 1:] - g[..., :-1]) / raster_time

    # Gradient is now sampled between k-space points whilst the slew rate is between gradient points
    if torch.is_tensor(k):
        sr = torch.zeros((*sr0.shape[:-1], sr0.shape[-1] + 1))
    else:
        sr = np.zeros((*sr0.shape[:-1], sr0.shape[-1] + 1))
    sr[..., 0] = sr0[..., 0]
    sr[..., 1:-1] = 0.5 * (sr0[..., :-1] + sr0[..., 1:])
    sr[..., -1] = sr0[..., -1]

    return g, sr


def align(**kwargs: Event | list[Event]) -> list[Event]:
    """Align passed events left or right by adjusting their delay.

    Example: `g1, g2, g3 = align(right=[g2, g3], center=g3)`"""

    if any([align not in ["left", "center", "right"] for align in kwargs]):
        raise ValueError("Invalid alignment spec.")

    alignments = []
    objects: list[Event] = []
    for arg_align, arg_objects in kwargs.items():
        if isinstance(arg_objects, list):
            event_list = cast(list[Event], arg_objects)
            alignments.extend([arg_align] * len(event_list))
            objects.extend(event_list)
        else:
            alignments.append(arg_align)
            objects.append(arg_objects)

    # Copy objects before adjusting delays - do not modify inputs
    objects = deepcopy(objects)
    dur = calc_duration(*objects)

    # Set new delays
    for object, alignment in zip(objects, alignments):
        # don't try to align soft delays - makes no sense
        if isinstance(object, SoftDelay):
            continue

        if alignment == "left":
            object.delay = 0
        elif alignment == "center":
            object.delay = (dur - object.duration) / 2
        elif alignment == "right":
            object.delay = dur - (object.duration - object.delay)
            if object.delay < 0:
                raise ValueError("Bug: align() attempts to set a negative delay")

    return objects
