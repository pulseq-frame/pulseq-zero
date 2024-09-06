from typing import Callable
from contextlib import contextmanager

# Fix for older pypulseq versions with newer numpy versions
import numpy as np
np.int = int
np.float = float
np.complex = complex

import torch
from . import facade

import MRzeroCore as mr0
import matplotlib.pyplot as plt


# Nam quantification phantom, should store as .npz instead
path = "W:/radiologie/mr-physik-data/Mitarbeiter/Zaiss_AG/Vorträge Gruppensitzung AG Zaiss/Konferenzbeiträge/2024_ESMRMB/Endres/"
data = torch.load(path + "nam.pt").detach()
data_B0 = torch.load(path + "nam_B0.pt").detach()
data_acs = torch.load(path + "image_acs.pt").detach()

PD = data_acs / data_acs.max()
PD[PD < 0.2] = 0
T1 = data[..., 0].clamp(0, 5)
T2 = data[..., 1].clamp(0, 1.5)
T2dash = data[..., 2].clamp(0, 0.5)
B1 = data[..., 3].clamp(0, 1.5)[None, ...]
D = data[..., 4].clamp(0, 4)
B0 = data_B0[..., 0].clamp(-50, 50)
coil_sens = torch.ones(1, *PD.shape)

mask = PD > 0.2
PD[~mask] = 0
T1[~mask] = 0
T2[~mask] = 0
T2dash[~mask] = 0
D[~mask] = 0
B0[~mask] = 0
B1[:, ~mask] = 0


phantom = mr0.VoxelGridPhantom(PD, T1, T2, T2dash, D, B0, B1, coil_sens, torch.tensor([0.2, 0.2, 0.008]))
# phantom = mr0.VoxelGridPhantom.brainweb("subject04.npz").interpolate(64, 64, 32).slices([15])
sim_data = phantom.build().cuda()
max_signal = phantom.PD.sum()


@contextmanager
def pp_intercept():
    facade.use_pulseqzero()
    try:
        yield
    finally:
        facade.use_pypulseq()


def simulate(seq_func: Callable[[], facade.Sequence], plot=False) -> torch.Tensor:
    with pp_intercept():
        seq = seq_func().to_mr0()

    for rep in seq:
        rep.gradm[:, 2] = 0

    graph = mr0.compute_graph(seq, sim_data, 1000, 1e-7)
    signal = mr0.execute_graph(graph, seq.cuda(), sim_data, 1e-3, 1e-4).cpu()
    reco = mr0.reco_adjoint(signal, seq.get_kspace(), (64, 64, 1), phantom.size)
    reco = reco[:, :, 0] / max_signal

    if plot:
        plt.figure(figsize=(7, 7), dpi=120)
        plt.subplot(211)
        plt.plot(signal.detach().real)
        plt.plot(signal.detach().imag)
        plt.grid()
        plt.subplot(223)
        plt.title("Magnitude")
        plt.imshow(reco.detach().abs().T, origin="lower", vmin=0, cmap="gray")
        plt.axis("off")
        plt.subplot(224)
        plt.title("Phase")
        plt.imshow(reco.detach().angle().T, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
        plt.axis("off")
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
