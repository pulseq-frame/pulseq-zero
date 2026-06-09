"""The meat of pulseq-zero: functions that mimic what's available in pypulseq
but internally store the inputs to forward them either to MR-zero or to
pypulseq. This allows pulseq-zero to provide functions like seq.write() where
the pulseq sequence is built on the fly, as well as seq.to_mr0() where the
sequence is converted to MR-zero while keeping the gradients intact."""

import numpy as np


def _n(x):
    """Detach to numpy: None -> None, scalar -> float, array/tensor -> np.ndarray."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x)
    if arr.shape == () or (arr.ndim == 1 and arr.shape[0] == 1):
        return float(arr.item())
    return arr
