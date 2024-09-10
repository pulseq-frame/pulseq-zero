from copy import copy, deepcopy
from pulseqzero.adapter import Opts, calc_duration

class Sequence:
    def __init__(self, system=None, use_block_cache=True):
        self.system = Opts.default if system is None else system
        self.definitions = {}
        self.blocks = []

    def __str__(self):
        return f"mr0 sequence adapter; ({len(self.blocks)}) blocks"

    def add_block(self, *args):
        self.blocks.append([copy(arg) for arg in args])

    def check_timing(self):
        return (True, [])

    def duration(self):
        duration = sum(calc_duration(*block) for block in self.blocks)
        num_blocks = len(self.blocks)
        event_count = len(filter(lambda b: len(b) > 0, self.blocks))
        return duration, num_blocks, event_count
    
    def get_definition(self, key):
        if key in self.definitions:
            return self.definitions[key]
        else:
            return ""

    def plot(self, label, show_blocks, save, time_range, time_disp, grad_disp, plot_now):
        pass

    def remove_duplicates(self, in_place=False):
        if in_place:
            return self
        else:
            return deepcopy(self)
    
    def set_definition(self, key, value):
        self.definitions[key] = value

    def test_report(self):
        return "No report generated in mr0 mode"

    def write(self, name, create_signature, remove_duplicates):
        if create_signature:
            return ""
        else:
            return None
