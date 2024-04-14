from .make_adc import Adc
from .make_block_pulse import BlockPulse
from .make_delay import Delay
from .make_extended_trapezoid import ExtendedTrapezoid
from .make_sinc_pulse import SincPulse
from .make_trapezoid import Trapezoid

class Sequence:
    def __init__(self, system=None) -> None:
        self.blocks = []

    def add_block(self, *args):
        self.blocks.append(args)

    def check_timing(self):
        return True, []
    
    def write(self, file_name):
        pass

    def plot(self):
        pass

    def to_mr0(self):
        for block in self.blocks:
            types = []
            for ev in block:
                if isinstance(ev, Adc):
                    types.append("adc")
                if isinstance(ev, BlockPulse):
                    types.append("rf_block")
                if isinstance(ev, Delay):
                    types.append("delay")
                if isinstance(ev, ExtendedTrapezoid):
                    types.append("trap_ex")
                if isinstance(ev, SincPulse):
                    types.append("rf_sinc")
                if isinstance(ev, Trapezoid):
                    types.append("trap")
            
            print(types)


# Copy the pypulseq importer logic here:
# we want an intermediate representation so we avoid appending to an mr0
# sequence all the time.
