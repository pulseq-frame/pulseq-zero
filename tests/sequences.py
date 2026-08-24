"""The sequences the fidelity tests are run on.

Two kinds:

* Four sequences written out here, each a plain PyPulseq script parameterized
  by the module it is executed against. They are small enough to simulate in a
  test and between them cover the event types the conversion has to get right:
  block pulses and refocusing pulses, trapezoids, delays, spoilers, ADCs, and
  a bipolar readout train.
* The unmodified upstream example scripts in ``pypulseq_examples/``, built
  through ``redirect.build()``.

Every builder takes the module to build against (``pypulseq`` or
``pulseqzero``) and returns a ``Sequence``.
"""

import numpy as np

from redirect import build

SYSTEM = dict(
    max_grad=28, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s",
    rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6,
)


def build_gre(pp, flip_deg=15.0, TE=10e-3, TR=30e-3, n=32, fov=0.2):
    """Spoiled gradient echo: block pulse, prewinder, readout, spoiler."""
    system = pp.Opts(**SYSTEM)
    seq = pp.Sequence(system=system)
    rf = pp.make_block_pulse(
        flip_angle=flip_deg * np.pi / 180, duration=1e-3, system=system,
        use="excitation",
    )
    gx = pp.make_trapezoid(channel="x", flat_area=n / fov, flat_time=2e-3, system=system)
    adc = pp.make_adc(num_samples=n, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=1e-3, system=system)
    gx_spoil = pp.make_trapezoid(channel="x", area=2 * n / fov, duration=1e-3, system=system)

    for i in range(n):
        seq.add_block(rf)
        gy = pp.make_trapezoid(channel="y", area=(i - n / 2) / fov, duration=1e-3, system=system)
        seq.add_block(gx_pre, gy)
        seq.add_block(pp.make_delay(TE - 1e-3))
        seq.add_block(gx, adc)
        gy_re = pp.make_trapezoid(channel="y", area=-(i - n / 2) / fov, duration=1e-3, system=system)
        seq.add_block(gx_spoil, gy_re)
        seq.add_block(pp.make_delay(TR - TE - 5e-3))

    seq.set_definition("FOV", [fov, fov, 8e-3])
    return seq


def build_se(pp, flip_deg=90.0, refoc_deg=180.0, TE=20e-3, TR=200e-3, n=32, fov=0.2):
    """Spin echo: exercises refocusing pulses and crushers."""
    system = pp.Opts(**SYSTEM)
    seq = pp.Sequence(system=system)
    rf = pp.make_block_pulse(
        flip_angle=flip_deg * np.pi / 180, duration=1e-3, system=system, use="excitation")
    rf_ref = pp.make_block_pulse(
        flip_angle=refoc_deg * np.pi / 180, duration=1e-3, phase_offset=np.pi / 2,
        system=system, use="refocusing")
    gx = pp.make_trapezoid(channel="x", flat_area=n / fov, flat_time=2e-3, system=system)
    adc = pp.make_adc(num_samples=n, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel="x", area=gx.area / 2, duration=1e-3, system=system)
    crush = pp.make_trapezoid(channel="z", area=4 * n / fov, duration=1e-3, system=system)

    for i in range(n):
        seq.add_block(rf)
        gy = pp.make_trapezoid(channel="y", area=(i - n / 2) / fov, duration=1e-3, system=system)
        seq.add_block(gx_pre, gy)
        seq.add_block(pp.make_delay(TE / 2 - 3e-3))
        seq.add_block(rf_ref, crush)
        seq.add_block(pp.make_delay(TE / 2 - 3e-3))
        seq.add_block(gx, adc)
        seq.add_block(pp.make_delay(TR - TE - 6e-3))

    seq.set_definition("FOV", [fov, fov, 8e-3])
    return seq


def build_epi(pp, flip_deg=90.0, TE=30e-3, n=32, fov=0.2):
    """Single-shot EPI: alternating readouts and phase blips."""
    system = pp.Opts(**dict(SYSTEM, max_grad=32, max_slew=130))
    seq = pp.Sequence(system=system)
    rf = pp.make_block_pulse(
        flip_angle=flip_deg * np.pi / 180, duration=1e-3, system=system, use="excitation")
    dwell = 5e-6
    gx = pp.make_trapezoid(channel="x", flat_area=n / fov, flat_time=n * dwell, system=system)
    gx_m = pp.make_trapezoid(channel="x", flat_area=-n / fov, flat_time=n * dwell, system=system)
    adc = pp.make_adc(num_samples=n, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=1e-3, system=system)
    gy_pre = pp.make_trapezoid(channel="y", area=-n / (2 * fov), duration=1e-3, system=system)
    gy_blip = pp.make_trapezoid(channel="y", area=1 / fov, duration=4e-4, system=system)

    seq.add_block(rf)
    seq.add_block(gx_pre, gy_pre)
    seq.add_block(pp.make_delay(TE - 1e-3))
    for i in range(n):
        seq.add_block(gx if i % 2 == 0 else gx_m, adc)
        if i < n - 1:
            seq.add_block(gy_blip)
    seq.set_definition("FOV", [fov, fov, 8e-3])
    return seq


def build_sinc_gre(pp, flip_deg=20.0, TE=8e-3, TR=25e-3, n=16, fov=0.2, slice_thickness=5e-3):
    """Slice-selective GRE: sinc pulse with slice-select and rephaser.

    The only builder here that exercises a shaped pulse together with the
    gradients that ``make_sinc_pulse`` returns.
    """
    system = pp.Opts(**SYSTEM)
    seq = pp.Sequence(system=system)
    rf, gz, gz_reph = pp.make_sinc_pulse(
        flip_angle=flip_deg * np.pi / 180, duration=2e-3, slice_thickness=slice_thickness,
        apodization=0.5, time_bw_product=4, system=system, return_gz=True, use="excitation",
    )
    gx = pp.make_trapezoid(channel="x", flat_area=n / fov, flat_time=2e-3, system=system)
    adc = pp.make_adc(num_samples=n, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, duration=1e-3, system=system)
    gz_spoil = pp.make_trapezoid(channel="z", area=4 / slice_thickness, duration=1e-3, system=system)

    for i in range(n):
        seq.add_block(rf, gz)
        gy = pp.make_trapezoid(channel="y", area=(i - n / 2) / fov, duration=1e-3, system=system)
        seq.add_block(gx_pre, gy, gz_reph)
        seq.add_block(pp.make_delay(TE - 2e-3))
        seq.add_block(gx, adc)
        seq.add_block(gz_spoil)
        seq.add_block(pp.make_delay(TR - TE - 6e-3))

    seq.set_definition("FOV", [fov, fov, slice_thickness])
    return seq


def _example(script):
    def builder(pp):
        return build(script, pp)

    builder.__name__ = script
    return builder


#: Sequences written for these tests; small enough to also simulate.
LOCAL = {
    "gre": build_gre,
    "se": build_se,
    "epi": build_epi,
    "sinc_gre": build_sinc_gre,
}

#: The unmodified upstream PyPulseq examples.
EXAMPLES = {
    name: _example(name) for name in [
        "write_gre",
        "write_haste",
        "write_radial_gre",
        "write_ute",
        "write_epi_se_rs",
        "write_tse",
    ]
}

ALL = {**LOCAL, **EXAMPLES}
