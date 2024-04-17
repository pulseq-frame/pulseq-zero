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
from contextlib import redirect_stdout


def simulate(seq_func: Callable[[], facade.Sequence], mode="full", plot=None) -> torch.Tensor:
    facade.use_pulseqzero()
    seq = seq_func().to_mr0()
    facade.use_pypulseq()

    # seq.plot_kspace_trajectory()

    phantom = mr0.VoxelGridPhantom.brainweb("subject04.npz")
    data = phantom.interpolate(64, 64, 32).slices([15]).build()

    if mode == "no T2":
        data.T2[:] = 1e9

    with redirect_stdout(None):
        graph = mr0.compute_graph(seq, data, 5000, 1e-4)
    signal = mr0.execute_graph(graph, seq, data, 0.01, 0.01)
    reco = mr0.reco_adjoint(signal, seq.get_kspace(), (64, 64, 1), phantom.size)[:, :, 0]

    # Automatic kspace reordering function in MRzeroCore would be great

    if plot:
        plt.figure()
        plt.suptitle(plot)
        plt.subplot(121)
        plt.imshow(reco.detach().abs().T, origin="lower", vmin=0)
        plt.subplot(122)
        plt.imshow(reco.detach().angle().T, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        plt.show()

    # TODO: simulate the mr0 sequence that is now contained in seq

    return reco


def optimize(loss_func: Callable[[torch.Tensor, bool], torch.Tensor], iters, param: torch.Tensor, lr):
    param.requires_grad = True
    params = [
        {"params": param, "lr": lr}
    ]
    optimizer = torch.optim.Adam(params)

    for i in range(iters):
        plot = i % 10 == 0 or i == iters - 1
        iter_str = f"Iteration {i+1} / {iters} - {param}"
        print(iter_str)
        facade.use_pulseqzero()
        loss = loss_func(param, iter_str if plot else None)
        facade.use_pypulseq()
        print(f" > Loss: {loss}")

        loss.backward()
        optimizer.step()
    
    return param
