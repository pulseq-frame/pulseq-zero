import torch
import numpy as np
import matplotlib.pyplot as plt
import MRzeroCore as mr0
import pulseqzero

# Import the sequences from their (slightly modified) scripts
from write_epi import main as build_epi
from write_gre import main as build_gre

data = mr0.VoxelGridPhantom.load("tests/moritz.npz").build()


def simulate(seq_func):
    with pulseqzero.mr0_mode():
        seq = seq_func().to_mr0()

    graph = mr0.compute_graph(seq, data)
    signal = mr0.execute_graph(graph, seq, data, print_progress=False)
    return mr0.reco_adjoint(signal, seq.get_kspace(), (64, 64, 1), (0.192, 0.192, 1))


def plot_results(start, best, loss_hist, param_hist):
    plt.figure(figsize=(7, 7), dpi=120)
    plt.subplot(221)
    plt.title("Magnitude")
    mr0.util.imshow(start.abs(), vmin=0, cmap="grey")
    plt.colorbar()
    plt.subplot(222)
    plt.title("Optimized")
    mr0.util.imshow(best.abs(), vmin=0, cmap="grey")
    plt.colorbar()
    plt.subplot(223)
    plt.title("Loss")
    plt.plot(loss_hist)
    plt.grid()
    plt.subplot(224)
    plt.title("Otim. param")
    plt.plot(param_hist)
    plt.grid()
    plt.show()


def epi_sim(TI):
    return simulate(lambda: build_epi(TI, False, False))


def gre_sim(flips):
    return simulate(lambda: build_gre(flips, False, False))


# ===================================
# Optimize the EPI sequence for FLAIR
# ===================================

TI = torch.tensor(1.0, requires_grad=True)
params = [{"params": TI, "lr": 0.02}]
optimizer = torch.optim.Adam(params)

loss_hist = []
TI_hist = []

start = epi_sim(TI)

for i in range(100):
    optimizer.zero_grad()
    reco = epi_sim(TI)

    # Calculate loss
    avg = reco.abs().mean()
    csf = reco[33, 37].abs()
    loss = csf - avg

    loss.backward()
    optimizer.step()

    loss_hist.append(loss.item())
    TI_hist.append(TI.item())
    print(f"{i+1} / 100: TI={TI.item()} | loss={loss.item()}")

best = epi_sim(TI_hist[np.argmin(loss_hist)])

# Plot the optimization result
plot_results(start, best, loss_hist, TI_hist)


# ==========================================
# Optimize the GRE sequence for low blurring
# ==========================================

flips = torch.full((64, ), 15 * np.pi / 180)

T1 = data.T1.clone()
data.T1[:] = 1e-6
target = gre_sim(flips)
data.T1 = T1

flips.requires_grad = True
params = [{"params": flips, "lr": 0.005}]
optimizer = torch.optim.Adam(params)

flip_hist = []
loss_hist = []

start = gre_sim(flips)

for i in range(100):
    optimizer.zero_grad()
    reco = gre_sim(flips)

    loss = ((reco.abs() - target.abs())**2).mean()
    loss.backward()
    optimizer.step()

    loss_hist.append(loss.item())
    flip_hist.append(flips.detach().numpy().copy())
    print(f"{i+1} / 100: loss={loss.item()}")

best = gre_sim(flip_hist[np.argmin(loss_hist)])

# Plot the optimization result
flip_hist = np.stack(flip_hist, 0) * 180 / np.pi
plot_results(start, best, loss_hist, flip_hist)
