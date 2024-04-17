from typing import Callable

# Fix for older pypulseq versions with newer numpy versions
import numpy as np
np.int = int
np.float = float
np.complex = complex

import torch
from . import loss
from . import facade

import MRzeroCore as mr0
import matplotlib.pyplot as plt


def simulate(seq_func: Callable[[], facade.Sequence], mode="full", plot=False) -> torch.Tensor:
    facade.use_pulseqzero()
    seq = seq_func().to_mr0()
    facade.use_pypulseq()

    # seq.plot_kspace_trajectory()

    phantom = mr0.VoxelGridPhantom.brainweb("subject04.npz")
    data = phantom.interpolate(64, 64, 32).slices([15]).build()

    if mode == "no T2":
        data.T2[:] = 1e9

    graph = mr0.compute_graph(seq, data, 5000, 1e-4)
    signal = mr0.execute_graph(graph, seq, data, 0.01, 0.01)
    reco = mr0.reco_adjoint(signal, seq.get_kspace(), (64, 64, 1), phantom.size)[:, :, 0]

    # Automatic kspace reordering function in MRzeroCore would be great

    if plot:
        plt.figure()
        plt.subplot(121)
        plt.imshow(reco.abs().T, origin="lower", vmin=0)
        plt.subplot(122)
        plt.imshow(reco.angle().T, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        plt.show()

    # TODO: simulate the mr0 sequence that is now contained in seq

    return reco


def optimize(loss_func, iters, param: torch.Tensor, lr) -> np.ndarray:
    return param.detach().numpy()
