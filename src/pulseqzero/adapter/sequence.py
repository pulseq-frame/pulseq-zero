from copy import copy, deepcopy
import MRzeroCore
import matplotlib.pyplot as plt
import numpy as np

from ..adapter import calc_duration, Opts
from ..adapter.delay import Delay
from ..adapter.adc import Adc
from ..adapter.pulses import Pulse
from ..adapter.grads import TrapGrad, FreeGrad
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
        return (True, [])

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

    def plot(self, show_blocks=False,
             time_range=None, time_unit="s", grad_unit="kHz/m"):
        # NOTE: This plotting function is not compatible with pypulseq!
        assert grad_unit == "kHz/m"

        time_factor = {
            "s": 1,
            "ms": 1e3,
            "us": 1e6
        }[time_unit]
        grad_factor = {
            "kHz/m": 1e-3,
            "mT/m": 1000 / self.system.gamma
        }[grad_unit]

        print("Definitions")
        width = max([len(key) for key in self.definitions])
        for (name, value) in self.definitions.items():
            print(f"> {name:<{width}} = {value!r}")

        dur, blocks, events = self.duration()
        print("Stats")
        print(f"> Duration    = {dur} s")
        print(f"> Block count = {blocks}")
        print(f"> Event count = {events}")

        t = 0
        grad_x = [[], []]
        grad_y = [[], []]
        grad_z = [[], []]
        rf = [[], []]

        for block in self.blocks:
            for event in block:
                if isinstance(event, Delay):
                    pass
                elif isinstance(event, Adc):
                    pass
                elif isinstance(event, Pulse):
                    time, amp = event._generate_shape()
                    rf[0] += (t + time).tolist() + [float("nan")]
                    rf[1] += amp.tolist() + [float("nan")]
                elif isinstance(event, TrapGrad):
                    if event.channel == "x":
                        grad = grad_x
                    elif event.channel == "y":
                        grad = grad_y
                    elif event.channel == "z":
                        grad = grad_z
                    else:
                        raise AttributeError(
                            f"Unexpected gradient channel: {event.channel!r}"
                        )

                    time = (time_factor * np.cumsum([
                        t + event.delay,
                        event.rise_time, event.flat_time, event.fall_time,
                        float("nan")
                    ])).tolist()
                    amp = grad_factor * event.amplitude

                    grad[0] += time
                    grad[1] += [0, amp, amp, 0, float("nan")]
                elif isinstance(event, FreeGrad):
                    pass

            t += float(calc_duration(*block))

        plt.subplot(312)
        plt.plot(rf[0], rf[1])
        plt.grid()
        if time_range:
            plt.xlim(time_range)
        plt.gca().tick_params(labelbottom=False)

        plt.subplot(313, sharex=plt.gca())
        plt.plot(grad_x[0], grad_x[1], label="x")
        plt.plot(grad_y[0], grad_y[1], label="y")
        plt.plot(grad_z[0], grad_z[1], label="z")
        plt.grid()
        plt.legend()
        if time_range:
            plt.xlim(time_range)

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

    # What we do all of this for:
    # To intercept pulseq calls and build an MR-zero sequence from it
    def to_mr0(self) -> MRzeroCore.Sequence:
        return seq_convert.convert(self)
