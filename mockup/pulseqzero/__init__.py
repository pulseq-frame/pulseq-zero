# Fix for older pypulseq versions with newer numpy versions
import numpy as np
np.int = int
np.float = float
np.complex = complex

import pypulseq as pypulseqfacade
import torch
from . import loss


def simulate(seq_func, mode="full", plot=False) -> torch.Tensor:
    return None


def optimize(loss_func, iters, param: torch.Tensor, lr) -> np.ndarray:
    return param.detach().numpy()
