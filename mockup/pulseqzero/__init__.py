from typing import Callable

# Fix for older pypulseq versions with newer numpy versions
import numpy as np
np.int = int
np.float = float
np.complex = complex

import torch
from . import loss
from . import facade


def simulate(seq_func: Callable[[], facade.Sequence], mode="full", plot=False) -> torch.Tensor:
    facade.use_pulseqzero()
    seq = seq_func().to_mr0()
    facade.use_pypulseq()

    seq.plot_kspace_trajectory()

    # TODO: simulate the mr0 sequence that is now contained in seq

    return None


def optimize(loss_func, iters, param: torch.Tensor, lr) -> np.ndarray:
    return param.detach().numpy()
