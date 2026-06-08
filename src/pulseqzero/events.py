"""Pulseq events from which sequence blocks are built.

Pypulseq creates `SimpleNameSpace`s with generated pulse shapes, computed areas
etc. directly in the various make_xxx calls. This contains lots of numpy code
that is not differentiable and also slow.

Pulseq-zero instead constructs dataclasses that contain all important properties
and compute shapes, durations etc. on request. The internal pulseq-zero sequence
structure can therefore be converted into an MR-zero sequence while keeping any
gradients from torch tensor inputs intact. For pulseq-specific functions like
plotting, a pulseq sequence can be reconstructed if necessary.

This file contains those dataclasses that are the internal pulseq-zero sequence.
"""

from __future__ import annotations
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Optional
from collections.abc import Callable
import numpy as np
import torch
import pypulseq as pp
from pypulseq import Opts
from .wrapper import _n


# A field that may carry a live torch tensor (differentiable) or a plain number.
Scalar = torch.Tensor | float
# Same but for array-likes (used for shim arrays).
Array = torch.Tensor | np.ndarray


@dataclass
class Label:
    label: str
    inc: bool
    value: int

    def to_pulseq(self, system: Opts):
        typ = "INC" if self.inc else "SET"
        pp.make_label(self.label, typ, self.value)


@dataclass
class RfPulse:
    flip_angle: Scalar
    freq_offset: Scalar
    phase_offset: Scalar
    delay: Scalar
    shape_dur: Scalar
    center: Scalar
    ringdown_time: float  # set via system param, do not modify afterwards
    use: str
    shim_array: Optional[Array]  # Martins pTx extension

    # Reconstruct pypulseq object from self - additional params stored in lambda.
    _pp_factory: Callable[[RfPulse, Opts], SimpleNamespace]

    @property
    def duration(self) -> Scalar:
        return self.delay + self.shape_dur + self.ringdown_time

    def to_pulseq(self, system: Opts) -> SimpleNamespace:
        # Pulses never generate gz / gzr: already split TrapGrad blocks
        return self._pp_factory(self, system)


# constructed in make_trapezoid
@dataclass
class TrapGrad:
    channel: str
    amplitude: Scalar
    rise_time: Scalar
    flat_time: Scalar
    fall_time: Scalar
    delay: Scalar

    @property
    def area(self) -> Scalar:
        return self.amplitude * (
            self.rise_time / 2 + self.flat_time + self.fall_time / 2
        )

    @property
    def flat_area(self) -> Scalar:
        return self.amplitude * self.flat_time

    @property
    def duration(self) -> Scalar:
        return self.delay + self.rise_time + self.flat_time + self.fall_time

    @property
    def first(self) -> float:
        return 0.0

    @property
    def last(self) -> float:
        return 0.0

    def to_pulseq(self, system: Opts) -> SimpleNamespace:
        return pp.make_trapezoid(
            channel=self.channel,
            amplitude=_n(self.amplitude),
            rise_time=_n(self.rise_time),
            flat_time=_n(self.flat_time),
            fall_time=_n(self.fall_time),
            delay=_n(self.delay),
            system=system,
        )


# constructed in make_arbitrary_grad
@dataclass
class ArbitraryGrad:
    channel: str
    waveform: Array
    delay: Scalar
    first: Scalar
    last: Scalar
    oversampling: bool
    _grad_raster: float

    @property
    def duration(self) -> Scalar:
        return self.delay + self.shape_dur

    @property
    def area(self) -> Scalar:
        if self.oversampling:
            return (self.waveform[::2] * self._grad_raster).sum()
        else:
            return (self.waveform * self._grad_raster).sum()

    @property
    def tt(self) -> Array:
        if self.oversampling:
            return np.arange(1, len(self.waveform) + 1) * 0.5 * self._grad_raster
        else:
            return (np.arange(len(self.waveform)) + 0.5) * self._grad_raster

    @property
    def shape_dur(self) -> Scalar:
        if self.oversampling:
            return (len(self.waveform) + 1) * 0.5 * self._grad_raster
        else:
            return len(self.waveform) * self._grad_raster

    def to_pulseq(self, system: Opts) -> SimpleNamespace:
        return pp.make_arbitrary_grad(
            channel=self.channel,
            waveform=_n(self.waveform),
            first=_n(self.first),
            last=_n(self.last),
            delay=_n(self.delay),
            system=system,
            oversampling=self.oversampling,
        )


# constructed in make_extended_trapezoid
@dataclass
class ExtTrapGrad:
    channel: str
    waveform: Array
    _times: Array

    @property
    def duration(self) -> Scalar:
        return self._times[-1]

    @property
    def delay(self) -> Scalar:
        return self._times[0]

    @delay.setter
    def delay(self, value: Scalar) -> None:
        times, value = _coerce(self._times, value)
        self._times = times - times[0] + value

    @property
    def tt(self) -> Array:
        return self._times - self.delay

    @property
    def shape_dur(self) -> Scalar:
        return self._times[-1] - self.delay

    @property
    def area(self) -> Scalar:
        times, waveform = _coerce(self._times, self.waveform)
        dt = times[1:] - times[:-1]
        mean_amp = 0.5 * (waveform[1:] + waveform[:-1])
        return (dt * mean_amp).sum()

    @property
    def first(self) -> Scalar:
        return self.waveform[0]

    @property
    def last(self) -> Scalar:
        return self.waveform[-1]

    def to_pulseq(self, system: Opts) -> SimpleNamespace:
        return pp.make_extended_trapezoid(
            channel=self.channel,
            amplitudes=_n(self.waveform),
            convert_to_arbitrary=False,
            times=_n(self._times),
            system=system,
        )


@dataclass
class Adc:
    num_samples: int
    dwell: Scalar
    delay: Scalar
    freq_offset: Scalar  # ignored by sim
    phase_offset: Scalar
    freq_ppm: Scalar
    phase_ppm: Scalar

    @property
    def duration(self) -> Scalar:
        return self.delay + self.num_samples * self.dwell

    def to_pulseq(self, system: Opts) -> SimpleNamespace:
        return pp.make_adc(
            num_samples=self.num_samples,
            delay=_n(self.delay),
            dwell=_n(self.dwell),
            freq_offset=_n(self.freq_offset),
            phase_offset=_n(self.phase_offset),
            system=system,
            freq_ppm=_n(self.freq_ppm),
            phase_ppm=_n(self.phase_ppm),
        )


@dataclass
class Delay:
    delay: Scalar
    _pp_event: SimpleNamespace  # could be a trigger or digital output pulse

    @property
    def duration(self) -> Scalar:
        return self.delay

    def to_pulseq(self, system: Opts) -> SimpleNamespace:
        return self._pp_event


@dataclass
class SoftDelay:
    hint: str
    numID: Optional[int]
    offset: Scalar
    factor: Scalar
    default_duration: Scalar  # Used if hint is not given or for computed dur.

    @property
    def duration(self) -> Scalar:
        return self.default_duration

    def to_pulseq(self, system: Opts) -> SimpleNamespace:
        return pp.make_soft_delay(
            self.hint,
            self.numID,
            float(self.offset),
            float(self.factor),
            float(self.default_duration),
        )


# Type for all pulseq-zero event types
Event = RfPulse | TrapGrad | ArbitraryGrad | ExtTrapGrad | Adc | Delay


# Helper used in some of the class properties - helps to stay numpy / torch agnostic
def _coerce(*operands: Array | Scalar) -> tuple:
    """Coerce operands to torch if any is a tensor. mixing them in arithmetic
    (numpy * tensor) raises, coercing keeps it working and preserves autograd."""
    if any(isinstance(x, torch.Tensor) for x in operands):
        return tuple(torch.as_tensor(x) for x in operands)
    return operands
