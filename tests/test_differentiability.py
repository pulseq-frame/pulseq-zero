"""Derivatives must reach every parameter the README documents as
differentiable, and must be the right ones.

Three levels, cheapest first:

1. straight-through rounding, on its own -- no simulation;
2. the flip angle that survives eager pulse-shape generation;
3. the derivative through construction, conversion and PDG simulation, against
   a central finite difference of the same loss.

Level 3 is where the pulseq-zero 1.0.1 defects showed: a numpy ``grad.tt``
reaching ``torch.abs``, a float64 ``pulse.angle`` and a ``torch.heaviside``
without a backward implementation each raised there rather than returning a
wrong number.
"""

import numpy as np
import pytest
import torch

import pulseqzero as pp
from pulseqzero.seq_convert import integrate_pulse

from conftest import simulate
from sequences import SYSTEM


# ---------------------------------------------------------------------------
# 1. Straight-through rounding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fn, reference",
    [(pp.round, np.round), (pp.ceil, np.ceil), (pp.floor, np.floor)],
    ids=["round", "ceil", "floor"],
)
@pytest.mark.parametrize("value", [1.4, 0.3, 2.0, -0.7])
def test_rounding_is_exact_forwards_and_identity_backwards(fn, reference, value):
    """Forward like the numpy function, backward like the identity - which is
    what makes a raster-aligned timing transport a derivative at all."""
    x = torch.tensor(value, requires_grad=True)
    y = fn(torch.sin(x))
    assert y.item() == pytest.approx(reference(np.sin(value)))

    y.backward()
    assert x.grad.item() == pytest.approx(np.cos(value), rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Flip angle through an eagerly generated envelope
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def system():
    return pp.Opts(**SYSTEM)


@pytest.mark.parametrize("factory_name", ["block", "sinc", "gauss"])
def test_simulated_flip_angle_matches_the_nominal_one(factory_name, system):
    """The shape is detached and the flip angle reattached as a scalar, so the
    integral over the pulse has to come back out as the flip angle that went
    in. A dead time counted into the integration window, or an integration
    window off by one raster step, shows up here."""
    factories = {
        "block": lambda: pp.make_block_pulse(
            flip_angle=np.pi / 2, duration=1e-3, system=system, use="excitation"),
        "sinc": lambda: pp.make_sinc_pulse(
            flip_angle=np.pi / 2, duration=2e-3, system=system,
            slice_thickness=5e-3, return_gz=False, use="excitation"),
        "gauss": lambda: pp.make_gauss_pulse(
            flip_angle=np.pi / 2, duration=2e-3, system=system,
            slice_thickness=5e-3, return_gz=False, use="excitation"),
    }
    rf = factories[factory_name]()
    if isinstance(rf, tuple):
        rf = rf[0]

    flip, _ = integrate_pulse(rf, 0.0, float(rf.duration))
    assert float(flip) == pytest.approx(np.pi / 2, rel=1e-5)


def test_flip_angle_carries_a_derivative(system):
    flip = torch.tensor(np.pi / 4, requires_grad=True)
    rf = pp.make_block_pulse(
        flip_angle=flip, duration=1e-3, system=system, use="excitation")
    angle, _ = integrate_pulse(rf, 0.0, float(rf.duration))
    angle.backward()
    # The integral is linear in the flip angle, so d(angle)/d(flip) == 1.
    assert flip.grad.item() == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 3. End-to-end derivatives against finite differences
# ---------------------------------------------------------------------------

def build_probe(flip, TE, prewinder_area, rf_phase, dwell, n=8, fov=0.2, reps=4):
    """A small GRE-like sequence exposing one parameter of each differentiable
    class: RF amplitude, timing, gradient amplitude, RF phase and ADC dwell."""
    system = pp.Opts(**SYSTEM)
    seq = pp.Sequence(system=system)
    rf = pp.make_block_pulse(flip_angle=flip, duration=1e-3, phase_offset=rf_phase,
                             system=system, use="excitation")
    # flat_time is fixed rather than n * dwell, so that `dwell` reaches the
    # loss through the ADC alone and not additionally through the
    # straight-through ceil() in the ramp time computation.
    gx = pp.make_trapezoid(channel="x", flat_area=n / fov, flat_time=200e-6, system=system)
    adc = pp.make_adc(num_samples=n, dwell=dwell, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel="x", area=prewinder_area, duration=1e-3, system=system)

    for i in range(reps):
        seq.add_block(rf)
        gy = pp.make_trapezoid(channel="y", area=(i - reps / 2) / fov, duration=1e-3,
                               system=system)
        seq.add_block(gx_pre, gy)
        seq.add_block(pp.make_delay(TE))
        seq.add_block(gx, adc)
        seq.add_block(pp.make_trapezoid(channel="z", area=400.0, duration=1e-3, system=system))
        seq.add_block(pp.make_delay(20e-3))
    return seq


def probe_loss(values, data):
    seq = build_probe(*values).to_mr0()
    sig = simulate(seq, data, max_state_count=200, min_state_mag=1e-4)
    # Phase-sensitive: a magnitude-only loss is invariant to the RF phase and
    # would give both autograd and the finite difference a zero derivative.
    return sig.real.sum() + 0.5 * (sig.abs() ** 2).sum()


#: name, value, finite-difference step, tolerance on the relative deviation.
#: The tolerances are loose because the reference is a finite difference of a
#: simulation, not an analytic derivative; they are far below what any of the
#: known defects produced (a wrong backward raises, a wrong forward moves the
#: derivative by tens of per cent or flips its sign).
PARAMETERS = [
    ("flip_angle", np.deg2rad(30.0), 1e-4, 2e-2),
    ("TE", 5e-3, 1e-6, 5e-2),
    ("prewinder_area", -80.0, 1e-2, 5e-2),
    ("rf_phase", 0.7, 1e-4, 2e-2),
    ("adc_dwell", 20e-6, 1e-9, 5e-2),
]
DEFAULTS = [value for _, value, _, _ in PARAMETERS]


@pytest.mark.parametrize("name, value, step, tolerance", PARAMETERS,
                         ids=[p[0] for p in PARAMETERS])
def test_derivative_matches_finite_difference(name, value, step, tolerance, phantom):
    """Each parameter is checked on its own, so that one whose backward pass is
    missing cannot be masked by the others."""
    index = [p[0] for p in PARAMETERS].index(name)

    values = [torch.tensor(float(v)) for v in DEFAULTS]
    values[index] = torch.tensor(float(value), requires_grad=True)
    probe_loss(values, phantom).backward()
    assert values[index].grad is not None, "no derivative reached the parameter"
    analytic = values[index].grad.item()

    def loss_at(x):
        shifted = [torch.tensor(float(v)) for v in DEFAULTS]
        shifted[index] = torch.tensor(float(x))
        with torch.no_grad():
            return probe_loss(shifted, phantom).item()

    finite_difference = (loss_at(value + step) - loss_at(value - step)) / (2 * step)
    deviation = abs(analytic - finite_difference) / max(abs(finite_difference), 1e-12)
    assert deviation < tolerance, (
        f"autograd {analytic:+.6e}, finite difference {finite_difference:+.6e} "
        f"(relative deviation {deviation:.2e})"
    )
