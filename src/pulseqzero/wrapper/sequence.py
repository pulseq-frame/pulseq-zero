from warnings import warn
from ..wrapper import _n
from copy import copy, deepcopy
import MRzeroCore
import pypulseq
import numpy as np
from pypulseq.utils.seq_plot import SeqPlot

from typing import Optional
from types import SimpleNamespace
from .. import calc_duration, Opts, seq_convert
from ..events import Event


class Sequence:
    def __init__(self, system: Optional[Opts] = None, use_block_cache=True):
        self.definitions = {}
        self.blocks: list[list[Event]] = []
        self.system = system if system else Opts.default
        self._pp_use_block_cache = use_block_cache

    def __str__(self):
        return f"mr0 sequence adapter; ({len(self.blocks)}) blocks"

    # =========================================================================
    # Ported functions
    # =========================================================================

    def add_block(self, *args: Event):
        self.blocks.append([copy(arg) for arg in args])

    def duration(self):
        duration = sum(calc_duration(*block) for block in self.blocks)
        num_blocks = len(self.blocks)
        event_count = sum(len(b) for b in self.blocks)
        return duration, num_blocks, event_count

    def get_definition(self, key):
        if key in self.definitions:
            return self.definitions[key]
        else:
            return ""

    def remove_duplicates(self, in_place=False):
        warn(
            "remove_duplicates() does nothing in pulseq-zero: It has no block "
            "cache (generates data lazily) that could be de-duplicated."
        )
        if in_place:
            return self
        else:
            return deepcopy(self)

    def set_definition(self, key, value):
        self.definitions[key] = value

    # =========================================================================
    # Non-differentiable functions - pypulseq pass-through
    # =========================================================================

    def adc_times(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        return self.to_pypulseq().adc_times(*args, **kwargs)

    def calculate_gradient_spectrum(
        self, *args, **kwargs
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        return self.to_pypulseq().calculate_gradient_spectrum(*args, **kwargs)

    def calculate_kspace(
        self, *args, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, list[float], list[float], np.ndarray]:
        return self.to_pypulseq().calculate_kspace(*args, **kwargs)

    def calculate_kspacePP(self, *args, **kwargs):
        raise NotImplementedError("calculate_kspacePP has been deprecated")

    def calculate_pns(self, *args, **kwargs):
        return self.to_pypulseq().calculate_pns(*args, **kwargs)

    def check_timing(self, *args, **kwargs) -> tuple[bool, list[SimpleNamespace]]:
        return self.to_pypulseq().check_timing(*args, **kwargs)

    def install(self, *args, **kwargs):
        return self.to_pypulseq().install(*args, **kwargs)

    def paper_plot(self, *args, **kwargs):
        return self.to_pypulseq().paper_plot(*args, **kwargs)

    def plot(self, *args, **kwargs) -> SeqPlot:
        return self.to_pypulseq().plot(*args, **kwargs)

    def read(self, *args, **kwargs):
        raise NotImplementedError("Cannot read .seq files in pulseq-zero")

    def rf_times(
        self, *args, **kwargs
    ) -> tuple[list[float], np.ndarray, list[float], np.ndarray, np.ndarray]:
        return self.to_pypulseq().rf_times(*args, **kwargs)

    def test_report(self, *args, **kwargs) -> str:
        return self.to_pypulseq().test_report(*args, **kwargs)

    def write(self, *args, **kwargs) -> str | None:
        return self.to_pypulseq().write(*args, **kwargs)

    # =========================================================================
    # PULSEQ-ZERO ADAPTER TO LIB CONVERSIONS
    #
    # The whole purpose of pulseq is to write differentiable code and only
    # swapping pypulseq -> pulseqzero imports. The wrapper then tracks all
    # calls and can convert the sequence to pypulseq or MR-zero. Pypulseq
    # conversion is done on-demand for applications like writing or plotting.
    # to_mr0 is the actual feature: convert to MR-zero while keeping gradients
    # intact, allowing sequence optimizations with pulseq-in-the-loop.
    # =========================================================================

    def to_pypulseq(self) -> pypulseq.Sequence:
        import warnings

        warnings.warn(
            "pulseqzero.Sequence: translating to PyPulseq — avoid calling "
            "this inside a hot loop.",
            stacklevel=2,
        )
        pp_seq = pypulseq.Sequence(
            system=self.system, use_block_cache=self._pp_use_block_cache
        )
        for block in self.blocks:
            pp_events = [ev.to_pulseq(self.system) for ev in block]
            pp_seq.add_block(*pp_events)
        for key, value in self.definitions.items():
            pp_seq.set_definition(key=key, value=_n(value))
        return pp_seq

    def to_mr0(
        self, samples_offres: int = 1, samples_slicesel: int = 1
    ) -> MRzeroCore.Sequence:
        return seq_convert.convert(self, samples_offres, samples_slicesel)
