from typing import Callable

# Fix for older pypulseq versions with newer numpy versions
import numpy as np
np.int = int
np.float = float
np.complex = complex

import torch
from . import facade

import MRzeroCore as mr0
import matplotlib.pyplot as plt


def simulate(seq_func: Callable[[], facade.Sequence], mode="full", plot=None) -> torch.Tensor:
    facade.use_pulseqzero()
    seq = seq_func().to_mr0()
    facade.use_pypulseq()

    for rep in seq:
        rep.gradm[:, 2] = 0

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
        plt.suptitle(plot)
        plt.subplot(221)
        plt.imshow(reco.detach().abs().T, origin="lower", vmin=0)
        plt.subplot(222)
        plt.imshow(reco.detach().angle().T, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        plt.subplot(212)
        plt.plot(signal.detach().real)
        plt.plot(signal.detach().imag)
        plt.grid()
        plt.show()

    return reco


def optimize(seq_func: Callable[[torch.Tensor], facade.Sequence], target: torch.Tensor, iters, param: torch.Tensor, lr):
    param.requires_grad = True
    params = [
        {"params": param, "lr": lr}
    ]
    optimizer = torch.optim.Adam(params)

    for i in range(iters):
        optimizer.zero_grad()
        plot = i % 10 == 0 or i == iters - 1
        iter_str = f"Iteration {i+1} / {iters} - {param}"
        print(iter_str)
        reco = simulate(lambda: seq_func(param), "full")
        loss = ((target - reco).abs()**2).mean()
        print(f" > Loss: {loss}")

        if plot:
            plt.figure()
            plt.suptitle(f"{loss} - {param}")
            plt.subplot(221)
            plt.imshow(reco.detach().abs().T, origin="lower", vmin=0)
            plt.colorbar()
            plt.subplot(222)
            plt.imshow(reco.detach().angle().T, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
            plt.subplot(223)
            plt.imshow(target.detach().abs().T, origin="lower", vmin=0)
            plt.colorbar()
            plt.subplot(224)
            plt.imshow(target.detach().angle().T, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
            plt.show()

        loss.backward()
        optimizer.step()
    
    return param.detach().numpy()
