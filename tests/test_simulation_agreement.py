"""The differentiable route into MR-zero must agree with the file route.

    (a) ``seq.to_mr0()``                              -- pulseq-zero
    (b) ``seq.write()`` + ``mr0.Sequence.import_file()`` -- the established,
        non-differentiable route, and the one whose code path a scanner
        interpreter also sees.

A discrepancy between (a) and (b) is a simulation/measurement mismatch of
exactly the kind the adapter exists to remove, so this is the correctness test
of the conversion. Five defects in pulseq-zero 1.0.1 (block-pulse dead time,
the RF envelope integration window, a numpy ``grad.tt`` reaching
``torch.abs``, a float64 ``pulse.angle``, a non-differentiable
``torch.heaviside``) were found by exactly this comparison.
"""

import MRzeroCore as mr0
import pytest
import torch

import pulseqzero

from conftest import simulate
from sequences import EXAMPLES, LOCAL

#: Both routes discretize the same waveforms slightly differently, so exact
#: agreement is not expected; anything above this is a conversion defect. The
#: five defects above each produced errors of 1e-2 or worse, or an exception.
TOLERANCE = 2e-3


#: ``write_ute`` is left out: both routes return exactly zero signal for it on
#: this phantom (a half-pulse UTE readout starting at the k-space centre with
#: no transverse magnetization left to sample), which makes the comparison
#: vacuous rather than passing. Its export is still covered by
#: ``test_export_fidelity.py``.
SIMULATED_EXAMPLES = sorted(set(EXAMPLES) - {"write_ute"})


def compare(seq, data, tmp_path, name):
    direct = torch.as_tensor(simulate(seq.to_mr0(), data).detach()).flatten()

    path = tmp_path / f"{name}.seq"
    seq.write(str(path))
    imported = torch.as_tensor(
        simulate(mr0.Sequence.import_file(str(path)), data).detach()
    ).flatten()

    assert direct.numel() == imported.numel(), (
        f"{direct.numel()} samples through to_mr0(), "
        f"{imported.numel()} through the .seq file"
    )
    reference = imported.abs().pow(2).mean().sqrt()
    assert reference > 1e-6, "no signal to compare - the sequence simulates to zero"
    return ((direct - imported).abs().pow(2).mean().sqrt() / reference).item()


@pytest.mark.parametrize("name", sorted(LOCAL))
def test_routes_agree(name, phantom, tmp_path):
    nrmse = compare(LOCAL[name](pulseqzero), phantom, tmp_path, name)
    assert nrmse < TOLERANCE, f"NRMSE {nrmse:.2e} between the two routes"


@pytest.mark.parametrize("name", SIMULATED_EXAMPLES)
def test_routes_agree_on_examples(name, phantom, tmp_path):
    """The same check on the unmodified upstream scripts."""
    nrmse = compare(EXAMPLES[name](pulseqzero), phantom, tmp_path, name)
    assert nrmse < TOLERANCE, f"NRMSE {nrmse:.2e} between the two routes"
