# Fix for older pypulseq versions with newer numpy versions
import numpy as np
np.int = int
np.float = float
np.complex = complex

import torch
from . import loss
from . import facade


def simulate(seq_func, mode="full", plot=False) -> torch.Tensor:
    # facade.use_real_pypulseq = False
    seq = seq_func()
    facade.use_real_pypulseq = True

    # TODO: return the mr0 sequence that is now contained in seq

    return None


def optimize(loss_func, iters, param: torch.Tensor, lr) -> np.ndarray:
    return param.detach().numpy()
