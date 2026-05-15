def _unsupported(name, workaround):
    def stub(*args, **kwargs):
        raise NotImplementedError(f"pulseqzero.{name} is not implemented. {workaround}")

    stub.__name__ = name
    return stub


make_adiabatic_pulse = _unsupported(
    "make_adiabatic_pulse",
    "Adiabatic pulses have no differentiable reimplementation yet. "
    "Workaround: design the pulse with `pypulseq.make_adiabatic_pulse` and "
    "wrap its signal via `pulseqzero.make_arbitrary_rf(signal=..., ...)`.",
)
sigpy_n_seq = _unsupported(
    "sigpy_n_seq",
    "sigpy-based pulse design is not supported. "
    "Workaround: build the pulse with sigpy + pypulseq and wrap the signal "
    "via `pulseqzero.make_arbitrary_rf(...)`.",
)
make_slr = _unsupported(
    "make_slr",
    "SLR pulse design is not supported. "
    "Workaround: design with sigpy / pypulseq then wrap via "
    "`pulseqzero.make_arbitrary_rf(...)`.",
)
make_sms = _unsupported(
    "make_sms",
    "SMS pulse design is not supported. "
    "Workaround: design with sigpy / pypulseq then wrap via "
    "`pulseqzero.make_arbitrary_rf(...)`.",
)
SigpyPulseOpts = _unsupported(
    "SigpyPulseOpts",
    "sigpy pulse options are not supported (see `make_slr` / `make_sms`).",
)
align = _unsupported(
    "align",
    "Timing alignment is not implemented in the adapter yet. "
    "Workaround: call `seq.to_pypulseq()` and use pypulseq's `align` on the "
    "translated sequence.",
)
calc_ramp = _unsupported(
    "calc_ramp",
    "calc_ramp is not implemented. "
    "Workaround: compute ramps manually or call `pypulseq.calc_ramp` via "
    "`seq.to_pypulseq()`.",
)
rotate = _unsupported(
    "rotate",
    "Gradient rotation is not implemented. "
    "Workaround: construct rotated gradients manually, or rotate after "
    "`seq.to_pypulseq()`.",
)
points_to_waveform = _unsupported(
    "points_to_waveform",
    "points_to_waveform is not implemented. "
    "Workaround: call `pypulseq.points_to_waveform` directly.",
)
traj_to_grad = _unsupported(
    "traj_to_grad",
    "traj_to_grad is not implemented. "
    "Workaround: call `pypulseq.traj_to_grad` directly.",
)
enable_trace = _unsupported(
    "enable_trace",
    "Event-creation tracing is a pypulseq diagnostic and is only meaningful "
    "inside pypulseq's sequence machinery. "
    "Workaround: call `pypulseq.enable_trace(...)` directly before/after "
    "`seq.to_pypulseq()` / `seq.write(...)`.",
)
disable_trace = _unsupported(
    "disable_trace",
    "Event-creation tracing is a pypulseq diagnostic and is only meaningful "
    "inside pypulseq's sequence machinery. "
    "Workaround: call `pypulseq.disable_trace()` directly.",
)
make_soft_delay = _unsupported(
    "make_soft_delay",
    "Soft delays are scanner-runtime adjustments and have no fixed duration "
    "to differentiate through. "
    "Workaround: use `pulseqzero.make_delay(d)` with `d` as your optimization "
    "parameter; for a true soft-delay block in the exported `.seq`, call "
    "`pypulseq.make_soft_delay(...)` on the result of `seq.to_pypulseq()`.",
)


# Silent no-op stubs (README ⚠️ "no-op"). Called by sequence scripts that
# ignore the return value; must not raise. Promote to real implementations
# (or to `_unsupported` raisers) when the behavior becomes load-bearing.
def calc_SAR(file):
    pass


def make_label(label, type, value):
    pass
