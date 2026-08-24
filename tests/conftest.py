"""Shared fixtures.

``tests/`` is added to ``sys.path`` so that the test modules can import
``sequences`` and ``redirect`` directly, and so that the example scripts, which
import each other's names from a flat directory, behave as they do upstream.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def phantom():
    """A 12-compartment analytic phantom. Nothing is downloaded."""
    import MRzeroCore as mr0
    import numpy as np

    rng = np.random.default_rng(0)
    n = 12
    pos = rng.uniform(-0.06, 0.06, (n, 3))
    pos[:, 2] = 0.0
    return mr0.CustomVoxelPhantom(
        pos=pos.tolist(),
        PD=rng.uniform(0.5, 1.0, n).tolist(),
        T1=rng.uniform(0.4, 1.8, n).tolist(),
        T2=rng.uniform(0.04, 0.15, n).tolist(),
        T2dash=rng.uniform(0.02, 0.05, n).tolist(),
        D=0.0,
        voxel_size=0.008,
    ).build()


def simulate(mr0_seq, data, max_state_count=2000, min_state_mag=1e-5):
    """Run PDG on `mr0_seq` and return the signal."""
    import MRzeroCore as mr0

    graph = mr0.compute_graph(mr0_seq, data, max_state_count, min_state_mag)
    return mr0.execute_graph(graph, mr0_seq, data, 1e-4, 1e-4, print_progress=False)
