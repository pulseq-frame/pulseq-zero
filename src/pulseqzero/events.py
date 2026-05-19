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
from collections.abc import Sequence, Callable
import numpy as np
import torch


# A field that may carry a live torch tensor (differentiable) or a plain number.
Scalar = torch.Tensor | float
# Same but for array-likes (used for shim arrays).
Array = torch.Tensor | np.ndarray | Sequence[Scalar]


@dataclass
class RfPulse:
    flip_angle: Scalar
    freq_offset: Scalar
    phase_offset: Scalar
    delay: Scalar
    shape_dur: Scalar
    ringdown_time: float  # set via system param, do not modify afterwards
    use: str
    shim_array: Optional[Array]  # Martins pTx extension

    # Reconstruct pypulseq object from self - additional params stored in lambda.
    _pp_factory: Callable[
        [RfPulse], SimpleNamespace | tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]
    ]

    @property
    def duration(self) -> Scalar:
        return self.delay + self.shape_dur + self.ringdown_time


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


@dataclass
class FreeGrad:
    channel: str
    waveform: torch.Tensor
    delay: Scalar
    tt: torch.Tensor
    shape_dur: Scalar
    first_waveform: Scalar
    last_waveform: Scalar

    @property
    def duration(self) -> Scalar:
        return self.delay + self.shape_dur

    @property
    def area(self) -> Scalar:
        return (
            0.5
            * (
                (self.tt[1:] - self.tt[:-1]) * (self.waveform[1:] + self.waveform[:-1])
            ).sum()
        )

    @property
    def first(self) -> Scalar:
        return self.first_waveform

    @property
    def last(self) -> Scalar:
        return self.last_waveform


@dataclass
class Adc:
    num_samples: int
    dwell: Scalar
    delay: Scalar
    freq_offset: Scalar  # ignored by sim
    phase_offset: Scalar

    @property
    def duration(self) -> Scalar:
        return self.delay + self.num_samples * self.dwell


@dataclass
class Delay:
    delay: Scalar

    @property
    def duration(self) -> Scalar:
        return self.delay
