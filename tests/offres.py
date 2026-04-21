import numpy as np
import torch
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import pulseqzero

# Import the sequences from their (slightly modified) scripts
from write_epi import main as build_epi
from write_gre import main as build_gre

with pulseqzero.mr0_mode():
    pp_seq = build_epi(1.5, False, False)
    # flips = torch.full((64, ), 15 * np.pi / 180)
    # pp_seq = build_gre(flips, False, False)
    # pp_seq.plot()
    seq = pp_seq.to_mr0(samples_slicesel=21)

time = np.cumsum([0] + [rep.event_time.sum() for rep in seq[1:-1]])
flips = [rep.pulse.angle for rep in seq[1:]]
plt.figure()
plt.title(f"{180/np.pi * np.sum(flips):.1f}°")
plt.plot(time * 1e3, [f * 180 / np.pi for f in flips])
plt.grid()
plt.ylabel("Flip [°]")
plt.xlabel("Time [ms]")
plt.show()
