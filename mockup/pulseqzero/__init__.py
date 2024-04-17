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


phantom = mr0.VoxelGridPhantom.brainweb("subject04.npz").interpolate(64, 64, 32).slices([15])
sim_data = phantom.build()
max_signal = phantom.PD.sum()


def simulate(seq_func: Callable[[], facade.Sequence], plot=False) -> torch.Tensor:
    facade.use_pulseqzero()
    seq = seq_func().to_mr0()
    facade.use_pypulseq()

    for rep in seq:
        rep.gradm[:, 2] = 0

    graph = mr0.compute_graph(seq, sim_data, 1000, 1e-3)
    signal = mr0.execute_graph(graph, seq, sim_data, 0.01, 0.01)
    reco = mr0.reco_adjoint(signal, seq.get_kspace(), (64, 64, 1), phantom.size)
    reco = reco[:, :, 0] / max_signal

    if plot:
        plt.figure()
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


def optimize(seq_func: Callable[[torch.Tensor], facade.Sequence],
             update_func: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
             iters: int, param: torch.Tensor, lr):
    param.requires_grad = True
    params = [
        {"params": param, "lr": lr}
    ]
    optimizer = torch.optim.Adam(params)

    for i in range(iters):
        optimizer.zero_grad()
        reco = simulate(lambda: seq_func(param))
        loss = update_func(param, reco, i)
        loss.backward()
        optimizer.step()
    
    return param.detach().numpy()
