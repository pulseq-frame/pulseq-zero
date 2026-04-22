from copy import copy, deepcopy
import MRzeroCore

from ..adapter import calc_duration, Opts
from . import seq_convert


class Sequence:
    def __init__(self, system: Opts = None, use_block_cache=True):
        self.definitions = {}
        self.blocks = []
        self.system = system if system else Opts.default

    def __str__(self):
        return f"mr0 sequence adapter; ({len(self.blocks)}) blocks"

    def add_block(self, *args):
        self.blocks.append([copy(arg) for arg in args])

    def check_timing(self):
        # Stub: the adapter cannot validate block timing on its own. Real
        # timing validation happens on export: either `seq.write()` runs
        # pypulseq's checks (it calls `check_timing` internally during write)
        # or users can call `seq.to_pypulseq().check_timing()` explicitly.
        # This stub keeps hot-loop callers (e.g. `build_tse` inside an Adam
        # loop) cheap and warning-free.
        return (True, [])

    def calculate_pns(self, *args, **kwargs):
        return self.to_pypulseq().calculate_pns(*args, **kwargs)

    def paper_plot(self, *args, **kwargs):
        return self.to_pypulseq().paper_plot(*args, **kwargs)

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

    def plot(self, *args, **kwargs):
        return self.to_pypulseq().plot(*args, **kwargs)

    def remove_duplicates(self, in_place=False):
        if in_place:
            return self
        else:
            return deepcopy(self)

    def set_definition(self, key, value):
        self.definitions[key] = value

    def test_report(self):
        return self.to_pypulseq().test_report()

    def write(self, name, create_signature=True, remove_duplicates=True):
        return self.to_pypulseq().write(
            name,
            create_signature=create_signature,
            remove_duplicates=remove_duplicates,
        )

    def to_pypulseq(self):
        import warnings
        import pypulseq
        from . import to_pypulseq as _to_pp

        warnings.warn(
            "pulseqzero.Sequence: translating to PyPulseq — avoid calling "
            "this inside a hot loop.",
            stacklevel=2,
        )
        pp_seq = pypulseq.Sequence(system=self.system)
        for block in self.blocks:
            pp_events = [_to_pp.event_to_pp(ev, self.system) for ev in block]
            pp_seq.add_block(*pp_events)
        for key, value in self.definitions.items():
            pp_seq.set_definition(key=key, value=_to_pp._n(value))
        return pp_seq

    # What we do all of this for:
    # To intercept pulseq calls and build an MR-zero sequence from it
    def to_mr0(self, samples_offres: int = 1, samples_slicesel: int = 1) -> MRzeroCore.Sequence:
        return seq_convert.convert(self, samples_offres, samples_slicesel)
