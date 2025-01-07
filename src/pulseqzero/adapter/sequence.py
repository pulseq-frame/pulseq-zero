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

    def plot(self, signal=None, reduced_plot=False, t=0,
             time_range=(0, float("inf")), time_unit="s", grad_unit="kHz/m",
             rf_max=None, adc_max=None):
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

        grad_x = [[], []]  # time, amp
        grad_y = [[], []]  # time, amp
        grad_z = [[], []]  # time, amp
        rf = [[], []]  # time, amp
        rf_phase = [[], []]  # pulse center, angle
        adc = [[], [], []]   # time, angle, signal
        nan = [float("nan")]  # used to split blocks in plotting
        if signal is not None:
            signal = signal.flatten()

        for block in self.blocks:
            block_dur = float(calc_duration(*block))
            if t + block_dur < time_range[0] or time_range[1] < t:
                t += block_dur
                continue

            for event in block:
                if isinstance(event, Delay):
                    pass
                elif isinstance(event, Adc):
                    adc[0] += (time_factor * (t + event.delay +
                               np.arange(event.num_samples) * event.dwell
                               )).tolist() + nan
                    adc[1] += [np.angle(np.exp(1j * event.phase_offset))] * event.num_samples + nan
                    if signal is not None:
                        adc[2] += signal[:event.num_samples].tolist() + nan
                        signal = signal[event.num_samples:]

                elif isinstance(event, Pulse):
                    time, amp = event._generate_shape()
                    rf[0] += (time_factor * (t + time)).tolist() + nan
                    rf[1] += amp.tolist() + nan
                    rf_phase[0].append(time_factor * (t + event.delay + event.shape_dur / 2))
                    rf_phase[1].append(np.angle(np.exp(1j * event.phase_offset)))
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
                        event.rise_time, event.flat_time, event.fall_time
                    ])).tolist()
                    amp = grad_factor * event.amplitude

                    grad[0] += time + nan
                    grad[1] += [0, amp, amp, 0] + nan
                elif isinstance(event, FreeGrad):
                    pass

            t += block_dur
        
        if rf_max is None:
            rf_max = 1.05 * float(max(rf[1]))
        if adc_max is None and signal is not None:
            adc_max = 1.05 * float(max(np.abs(signal)))
        
        plt.subplot(211 if reduced_plot else 411)
        plt.plot(rf[0], rf[1], c="k")
        plt.ylim(-0.05 * rf_max, 1.05 * rf_max)
        plt.grid()
        plt.ylabel("RF [Hz]")
        plt.gca().tick_params(labelbottom=False)

        if reduced_plot and signal is not None:
            ticks, _ = plt.yticks()
            ax = plt.gca().twinx()
            ax.plot(adc[0], np.abs(adc[2]), label="abs", c="tab:red")
            ax.set_yticks(ticks * adc_max / rf_max)
            ax.set_ylim(-0.05 * adc_max, 1.05 * adc_max)
            ax.grid(False)
            ax.tick_params("y", colors="tab:red")
            ax.spines["right"].set_color("tab:red")
            ax.set_ylabel("Signal [a.u.]", color="tab:red")

        plt.subplot(212 if reduced_plot else 412, sharex=plt.gca())
        plt.plot(grad_x[0], grad_x[1], label="x")
        plt.plot(grad_y[0], grad_y[1], label="y")
        plt.plot(grad_z[0], grad_z[1], label="z")
        plt.grid()
        plt.legend()
        plt.ylabel(f"Gradient [{grad_unit}]")
        if not reduced_plot:
            plt.gca().tick_params(labelbottom=False)
        else:
            plt.xlabel(f"Time [{time_unit}]")
            return

        plt.subplot(413, sharex=plt.gca())
        plt.plot(adc[0], np.zeros(len(adc[0])), "rx")
        if signal is not None:
            plt.plot(adc[0], np.abs(adc[2]), label="abs")
            plt.plot(adc[0], np.real(adc[2]), label="real")
            plt.plot(adc[0], np.imag(adc[2]), label="imag")
        plt.legend()
        plt.grid()
        plt.gca().tick_params(labelbottom=False)

        plt.subplot(414, sharex=plt.gca())
        plt.plot(rf_phase[0], rf_phase[1], ".", label="pulse")
        plt.plot(adc[0], adc[1], "rx", label="adc")
        plt.ylim(-1.1 * np.pi, 1.1 * np.pi)
        plt.yticks(
            [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            ["$-180°$", "$-90°$", "$0°$", "$90°$", "$180°$"]
        )
        plt.grid()
        plt.legend()
        plt.xlabel(f"Time [{time_unit}]")

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
