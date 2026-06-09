def _unsupported(name, workaround):
    def stub(*args, **kwargs):
        raise NotImplementedError(f"pulseqzero.{name} is not implemented. {workaround}")

    stub.__name__ = name
    return stub


calc_SAR = _unsupported(
    "calc_SAR",
    "Built-in SAR computation has been removed; use PySar4seq"
    "(https://github.com/imr-framework/sar4seq/tree/PySar4seq) instead.",
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
SigpyPulseOpts = _unsupported(
    "SigpyPulseOpts",
    "sigpy pulse options are not supported (see `make_slr` / `make_sms`).",
)
sigpy_n_seq = _unsupported(
    "sigpy_n_seq",
    "sigpy-based pulse design is not supported. "
    "Workaround: build the pulse with sigpy + pypulseq and wrap the signal "
    "via `pulseqzero.make_arbitrary_rf(...)`.",
)
make_adiabatic_pulse = _unsupported(
    "make_adiabatic_pulse",
    "Adiabatic pulses have no differentiable reimplementation yet. "
    "Workaround: design the pulse with `pypulseq.make_adiabatic_pulse` and "
    "wrap its signal via `pulseqzero.make_arbitrary_rf(signal=..., ...)`.",
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
