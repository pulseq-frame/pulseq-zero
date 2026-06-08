"""Completeness test for the pulseq-zero adapter (post-mode-switch API).

Pulseq-zero 1.x removed the mode-switching facade: ``import pulseqzero as pp``
is a drop-in for ``import pypulseq as pp``, and the *same* ``Sequence`` object
supports both ``seq.write(...)`` and ``seq.to_mr0()`` unconditionally - no
``pp_impl`` indirection, no ``with pulseqzero.mr0_mode():`` block.

This script verifies that contract end-to-end:

1. ``build_broad_sequence`` builds the same broad sequence against both
   ``pypulseq`` and ``pulseqzero`` using only the entries documented as native
   (READMEs section 4, "PyPulseq coverage" table).
2. ``probe_api`` exercises every other entry point in that table in isolation
   and records its status (ok / not_implemented / missing / error).
3. ``probe_exports`` calls the export forwarders on the *built* sequence:
   ``write``, ``to_pypulseq``, ``test_report``, and (pulseq-zero only)
   ``to_mr0``.
4. ``print_support_matrix`` prints a side-by-side ``pypulseq | pulseqzero``
   matrix and writes it to a markdown file.
"""

import numpy as np
import pypulseq
import pulseqzero


def try_call(fn, *args, **kwargs):
    """Run ``fn`` and classify the outcome.

    Returns ``(status, value_or_None, error_message_or_None)`` where ``status``
    is one of ``"ok"``, ``"not_implemented"``, ``"missing"``, ``"error"``.
    """
    try:
        return "ok", fn(*args, **kwargs), None
    except NotImplementedError as exc:
        return "not_implemented", None, str(exc)
    except AttributeError as exc:
        return "missing", None, str(exc)
    except Exception as exc:
        return "error", None, f"{type(exc).__name__}: {exc}"


def _summarize(val):
    if val is None:
        return "None"
    if hasattr(val, "shape"):
        return f"<{type(val).__name__} shape={tuple(np.shape(val))}>"
    if isinstance(val, (list, tuple)):
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, dict):
        return f"<dict len={len(val)}>"
    if isinstance(val, (int, float, np.floating, np.integer)):
        return f"{float(val):.6g}"
    if isinstance(val, str):
        return repr(val[:40])
    return f"<{type(val).__name__}>"


# ---------------------------------------------------------------------------
# Sequence construction (uses only the entries documented as native in both
# backends, so the same body runs against pypulseq and pulseq-zero).
# ---------------------------------------------------------------------------

def build_broad_sequence(pp, out_seq_path=None, do_plot=False, do_write=False):
    system = pp.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
        grad_raster_time=10e-6,
    )

    seq = pp.Sequence(system=system)
    # NB: only numeric definitions are set here. pulseq-zero's
    # `Sequence.to_pypulseq` passes every definition value through `_n` which
    # coerces to float, so a string definition (e.g. `set_definition("Name",
    # "...")`) blows up at export. The script needs `to_pypulseq` /
    # `write` / `test_report` to work in order to probe them; if the
    # adapter starts forwarding strings unchanged, restoring a string
    # definition here is a one-line change.
    seq.set_definition("FOV", [0.22, 0.22, 0.005])
    seq.set_definition("matrix", [64, 64, 1])
    seq.set_definition("TE", 0.03)
    seq.set_definition("TR", 0.2)

    # RF pulses - all four factories.
    rf_block = pp.make_block_pulse(
        flip_angle=np.deg2rad(15),
        duration=1e-3,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    rf_sinc, gz_sinc, gzr_sinc = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(90),
        duration=3e-3,
        slice_thickness=5e-3,
        apodization=0.5,
        time_bw_product=4,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use="excitation",
    )
    rf_gauss, gz_gauss, gzr_gauss = pp.make_gauss_pulse(
        flip_angle=np.deg2rad(30),
        duration=2e-3,
        slice_thickness=5e-3,
        apodization=0.25,
        time_bw_product=4,
        delay=system.rf_dead_time,
        system=system,
        return_gz=True,
        use="excitation",
    )
    rf_arb = pp.make_arbitrary_rf(
        signal=np.hanning(128).astype(float),
        flip_angle=np.deg2rad(20),
        delay=system.rf_dead_time,
        dwell=10e-6,
        system=system,
        use="excitation",
    )

    # Trapezoids in all three creation styles. We pass an explicit raster-
    # aligned `rise_time` because pulseq-zero's `make_trapezoid` does *not*
    # round rise/fall times to the gradient raster (intentional: rounding
    # would kill autograd). The TSE demo does the same - see
    # demo/write_tse.py's `dG = 250e-6`. Without this, `seq.write()` fails
    # because pypulseq's `check_timing` rejects unaligned timings.
    rise = 50e-6  # 5 raster ticks at grad_raster_time=10e-6
    gx_area = pp.make_trapezoid(
        channel="x", area=4.0, duration=2e-3, rise_time=rise, system=system,
    )
    gy_flat = pp.make_trapezoid(
        channel="y", flat_area=8.0, flat_time=2e-3, rise_time=rise, system=system,
    )
    gz_amp = pp.make_trapezoid(
        channel="z", amplitude=6.0, duration=1.5e-3, rise_time=rise, system=system,
    )

    # Extended trapezoid and arbitrary gradient.
    ext_times = np.array([0.0, 0.5e-3, 1.2e-3, 2.0e-3])
    ext_amps = np.array(
        [0.0, 0.8 * system.max_grad, -0.6 * system.max_grad, 0.0]
    )
    gx_ext = pp.make_extended_trapezoid(
        channel="x", times=ext_times, amplitudes=ext_amps, system=system
    )

    t = np.arange(0, 1e-3, system.grad_raster_time)
    arb_wave = 0.2 * system.max_grad * np.sin(2 * np.pi * t / t[-1])
    gy_arb = pp.make_arbitrary_grad(channel="y", waveform=arb_wave, system=system)

    # Composite/derived gradients. `add_gradients` and `split_gradient` are
    # *not* added as sequence blocks here: they're verified standalone by the
    # probe phase. Inlining them in the broad seq would propagate
    # non-raster-aligned times into the exporter (a real pulseq-zero
    # limitation that should not stop the rest of the matrix from running).
    gx_scaled = pp.scale_grad(gx_area, 0.5)

    adc = pp.make_adc(num_samples=100, dwell=20e-6, delay=20e-6, system=system)
    d_short = pp.make_delay(500e-6)
    d_long = pp.make_delay(2e-3)
    gx_readout = pp.make_trapezoid(
        channel="x", flat_area=32.0, flat_time=2e-3, rise_time=rise, system=system,
    )

    # Build sequence block-by-block. The shaped RF pulses (rf_sinc /
    # rf_gauss) are added *without* their accompanying slice-select gz - the
    # gz returned by `make_*_pulse(return_gz=True)` has rise/fall times that
    # pulseq-zero does not round to the raster, so co-adding them would
    # break `seq.write()`. Production scripts (see demo/write_tse.py) follow
    # the same pattern: pull amplitude/duration from the returned gz and
    # rebuild a raster-aligned trapezoid via `make_trapezoid`.
    seq.add_block(rf_block)
    seq.add_block(d_short)
    seq.add_block(rf_sinc)
    seq.add_block(d_short)
    seq.add_block(gx_area, gy_flat)
    seq.add_block(gz_amp)
    seq.add_block(gx_ext)
    seq.add_block(gy_arb)
    seq.add_block(gx_readout, adc)
    seq.add_block(gx_scaled)
    seq.add_block(rf_gauss)
    seq.add_block(rf_arb)
    seq.add_block(d_long)

    timing_ok, timing_report = seq.check_timing()

    durations = {
        "rf_block": float(pp.calc_duration(rf_block)),
        "rf_sinc+gz": float(pp.calc_duration(rf_sinc, gz_sinc)),
        "gx_area": float(pp.calc_duration(gx_area)),
        "gy_flat": float(pp.calc_duration(gy_flat)),
        "adc": float(pp.calc_duration(adc)),
    }

    if do_plot:
        seq.plot(time_range=[0, 0.03], plot_now=False)
    if do_write and out_seq_path:
        seq.write(out_seq_path)

    return {
        "seq": seq,
        "system": system,
        "rf_block": rf_block,
        "rf_sinc": rf_sinc,
        "rf_gauss": rf_gauss,
        "rf_arb": rf_arb,
        "gx_area": gx_area,
        "gy_flat": gy_flat,
        "gz_amp": gz_amp,
        "gx_ext": gx_ext,
        "gy_arb": gy_arb,
        "arb_wave": arb_wave,
        "timing_ok": timing_ok,
        "timing_report": timing_report,
        "durations": durations,
        "out_seq_path": out_seq_path if do_write else None,
    }


# ---------------------------------------------------------------------------
# Per-entry-point probes. Each entry lists (name, expected_pulseqzero_status,
# fn(pp, ctx)). The pypulseq side calls the same fn; the matrix then shows
# the actual behavior on each backend.
# ---------------------------------------------------------------------------

def _native_probes():
    def s(ctx):
        return ctx["system"]

    return [
        ("Opts", "ok",
            lambda pp, ctx: pp.Opts(max_grad=28, grad_unit="mT/m")),
        ("Sequence.__init__", "ok",
            lambda pp, ctx: pp.Sequence(system=s(ctx))),
        ("Sequence.add_block", "ok",
            lambda pp, ctx: (lambda q: q.add_block(pp.make_delay(1e-3)) or q)(
                pp.Sequence(system=s(ctx))
            )),
        ("Sequence.set_definition/get_definition", "ok",
            lambda pp, ctx: (lambda q: (
                q.set_definition("Probe", 42), q.get_definition("Probe")
            )[1])(pp.Sequence(system=s(ctx)))),
        ("Sequence.duration", "ok",
            lambda pp, ctx: ctx["seq"].duration()),
        ("Sequence.__str__", "ok",
            lambda pp, ctx: str(ctx["seq"])),
        ("Sequence.remove_duplicates", "ok",
            lambda pp, ctx: ctx["seq"].remove_duplicates()),
        ("Sequence.check_timing", "ok",
            lambda pp, ctx: ctx["seq"].check_timing()),
        ("calc_duration", "ok",
            lambda pp, ctx: pp.calc_duration(ctx["gx_area"])),
        ("get_supported_labels", "ok",
            lambda pp, ctx: pp.get_supported_labels()),
        ("make_adc", "ok",
            lambda pp, ctx: pp.make_adc(num_samples=64, dwell=10e-6, system=s(ctx))),
        ("make_delay", "ok",
            lambda pp, ctx: pp.make_delay(1e-3)),
        ("make_trigger", "ok",
            lambda pp, ctx: pp.make_trigger("physio1", duration=100e-6, system=s(ctx))),
        ("make_digital_output_pulse", "ok",
            lambda pp, ctx: pp.make_digital_output_pulse(
                "osc0", duration=100e-6, system=s(ctx)
            )),
        ("make_trapezoid", "ok",
            lambda pp, ctx: pp.make_trapezoid(
                channel="x", area=1.0, duration=1e-3, system=s(ctx)
            )),
        ("make_extended_trapezoid", "ok",
            lambda pp, ctx: pp.make_extended_trapezoid(
                channel="x",
                times=np.array([0.0, 0.5e-3, 1.0e-3]),
                amplitudes=np.array([0.0, 0.5 * s(ctx).max_grad, 0.0]),
                system=s(ctx),
            )),
        ("make_arbitrary_grad", "ok",
            lambda pp, ctx: pp.make_arbitrary_grad(
                channel="y", waveform=ctx["arb_wave"], system=s(ctx)
            )),
        ("add_gradients", "ok",
            lambda pp, ctx: pp.add_gradients(
                [
                    ctx["gx_area"],
                    pp.make_trapezoid(
                        channel="x", area=2.0, duration=2e-3, system=s(ctx)
                    ),
                ],
                system=s(ctx),
            )),
        ("scale_grad", "ok",
            lambda pp, ctx: pp.scale_grad(ctx["gx_area"], 0.5)),
        ("split_gradient", "ok",
            lambda pp, ctx: pp.split_gradient(ctx["gx_area"], system=s(ctx))),
        ("make_block_pulse", "ok",
            lambda pp, ctx: pp.make_block_pulse(
                flip_angle=np.deg2rad(15),
                duration=1e-3,
                delay=s(ctx).rf_dead_time,
                system=s(ctx),
                use="excitation",
            )),
        ("make_sinc_pulse", "ok",
            lambda pp, ctx: pp.make_sinc_pulse(
                flip_angle=np.deg2rad(90),
                duration=3e-3,
                slice_thickness=5e-3,
                apodization=0.5,
                time_bw_product=4,
                delay=s(ctx).rf_dead_time,
                system=s(ctx),
                return_gz=True,
                use="excitation",
            )),
        ("make_gauss_pulse", "ok",
            lambda pp, ctx: pp.make_gauss_pulse(
                flip_angle=np.deg2rad(30),
                duration=2e-3,
                slice_thickness=5e-3,
                apodization=0.25,
                time_bw_product=4,
                delay=s(ctx).rf_dead_time,
                system=s(ctx),
                return_gz=True,
                use="excitation",
            )),
        ("make_arbitrary_rf", "ok",
            lambda pp, ctx: pp.make_arbitrary_rf(
                signal=np.hanning(128).astype(float),
                flip_angle=np.deg2rad(20),
                delay=s(ctx).rf_dead_time,
                dwell=10e-6,
                system=s(ctx),
                use="excitation",
            )),
        # make_extended_trapezoid_area is copied verbatim from pypulseq -
        # works, but not differentiable. README marks it as such.
        ("make_extended_trapezoid_area", "ok",
            lambda pp, ctx: pp.make_extended_trapezoid_area(
                channel="x",
                area=1.0,
                grad_start=0.0,
                grad_end=0.0,
                system=s(ctx),
            )),
        # points_to_waveform is implemented in pulseq-zero's wrapper layer
        # (the README table is out of date - the wrapper version overrides
        # the not_implemented stub).
        ("points_to_waveform", "ok",
            lambda pp, ctx: pp.points_to_waveform(
                amplitudes=np.array([0.0, 0.8 * s(ctx).max_grad, 0.0]),
                grad_raster_time=s(ctx).grad_raster_time,
                times=np.array([0.0, 0.5e-3, 1.0e-3]),
            )),
        # Numeric stubs - pulseq-zero returns approximations.
        ("calc_rf_bandwidth (stub)", "ok",
            lambda pp, ctx: pp.calc_rf_bandwidth(ctx["rf_sinc"])),
        ("calc_rf_center (stub)", "ok",
            lambda pp, ctx: pp.calc_rf_center(ctx["rf_sinc"])),
        # make_label - pypulseq returns a SimpleNamespace; pulseq-zero
        # returns its own Label dataclass (both should not raise).
        ("make_label", "ok",
            lambda pp, ctx: pp.make_label(label="LIN", type="SET", value=1)),
    ]


def _not_implemented_probes():
    """Entries pulseq-zero deliberately raises NotImplementedError on.

    Plain pypulseq still implements most of these, so the matrix highlights
    where pulseq-zero diverges from upstream by design.
    """
    def s(ctx):
        return ctx["system"]

    return [
        ("make_adiabatic_pulse", "not_implemented",
            lambda pp, ctx: pp.make_adiabatic_pulse(
                flip_angle=np.pi, duration=8e-3, system=s(ctx)
            )),
        ("sigpy_n_seq", "not_implemented",
            lambda pp, ctx: pp.sigpy_n_seq(flip_angle=np.pi)),
        ("make_slr", "not_implemented",
            lambda pp, ctx: pp.make_slr()),
        ("make_sms", "not_implemented",
            lambda pp, ctx: pp.make_sms()),
        ("SigpyPulseOpts", "not_implemented",
            lambda pp, ctx: pp.SigpyPulseOpts()),
        ("align", "ok",
            lambda pp, ctx: pp.align(
                right=[ctx["gx_area"], ctx["gy_flat"], ctx["gz_amp"]]
            )),
        ("calc_ramp", "ok",
            lambda pp, ctx: pp.calc_ramp(
                k0=np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                k_end=np.array([[1e-3, 2e-3], [0.0, 0.0], [0.0, 0.0]]),
                system=s(ctx),
            )),
        ("rotate", "ok",
            lambda pp, ctx: pp.rotate(
                ctx["gx_area"], ctx["gy_flat"], angle=np.pi / 8, axis="z"
            )),
        ("traj_to_grad", "ok",
            lambda pp, ctx: pp.traj_to_grad(
                np.linspace(0.0, 1.0, 16), raster_time=s(ctx).grad_raster_time
            )),
        ("calc_SAR", "not_implemented",
            lambda pp, ctx: pp.calc_SAR(ctx["seq"])),
        ("make_soft_delay", "ok",
            lambda pp, ctx: pp.make_soft_delay(
                hint="user", numID=1, default_duration=1e-3
            )),
        ("enable_trace", "not_implemented",
            lambda pp, ctx: pp.enable_trace()),
        ("disable_trace", "not_implemented",
            lambda pp, ctx: pp.disable_trace()),
        ("split_gradient_at", "ok",
            lambda pp, ctx: pp.split_gradient_at(
                ctx["gx_area"], time_point=1e-3, system=s(ctx)
            )),
    ]


def _export_probes():
    """Export forwarders - run after the broad sequence is built."""

    def to_pypulseq_safe(seq):
        # Only pulseq-zero defines to_pypulseq(); pypulseq Sequence does not.
        return seq.to_pypulseq()

    return [
        ("Sequence.write", lambda pp, ctx: ctx["seq"].write(ctx["out_seq_path"])),
        ("Sequence.to_pypulseq", lambda pp, ctx: to_pypulseq_safe(ctx["seq"])),
        ("Sequence.test_report", lambda pp, ctx: ctx["seq"].test_report()),
        # Only pulseq-zero exposes to_mr0; pypulseq Sequence does not.
        ("Sequence.to_mr0", lambda pp, ctx: ctx["seq"].to_mr0()),
    ]


def probe_module(pp, label, out_seq_path):
    """Build a broad seq with `pp`, then probe every API entry point."""
    print(f"\n=== probing {label} ===")
    ctx = build_broad_sequence(
        pp, out_seq_path=out_seq_path, do_plot=False, do_write=False
    )
    ctx["out_seq_path"] = out_seq_path

    results = {}
    for name, expected, fn in _native_probes() + _not_implemented_probes():
        status, val, err = try_call(fn, pp, ctx)
        results[name] = {
            "expected": expected,
            "status": status,
            "summary": _summarize(val),
            "error": err,
        }

    for name, fn in _export_probes():
        status, val, err = try_call(fn, pp, ctx)
        # Sequence.write returns None on success - that's still "ok".
        results[name] = {
            "expected": "ok",
            "status": status,
            "summary": _summarize(val),
            "error": err,
        }

    print(f"  built broad seq with {ctx['seq'].duration()[1]} blocks; "
          f"check_timing={ctx['timing_ok']}")
    return results, ctx


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_STATUS_SYMBOL = {
    "ok": "OK",
    "not_implemented": "NotImpl",
    "missing": "missing",
    "error": "ERROR",
}


def print_support_matrix(pp_results, pz_results):
    print("\n=== support matrix ===")
    header = f"{'api entry':<48} | {'pypulseq':<10} | {'pulseqzero':<14} | expected"
    print(header)
    print("-" * len(header))

    all_keys = sorted(set(pp_results) | set(pz_results))
    rows = []
    for key in all_keys:
        pp_st = pp_results.get(key, {}).get("status", "missing")
        pz_st = pz_results.get(key, {}).get("status", "missing")
        expected = pz_results.get(key, {}).get(
            "expected", pp_results.get(key, {}).get("expected", "?")
        )
        rows.append((key, pp_st, pz_st, expected))
        ok_pz = pz_st == expected
        mark = "" if ok_pz else "  <-- MISMATCH"
        print(
            f"{key:<48} | {_STATUS_SYMBOL[pp_st]:<10} | "
            f"{_STATUS_SYMBOL[pz_st]:<14} | {expected}{mark}"
        )
    return rows


def print_probe_details(label, results):
    print(f"\n=== {label} probe details ===")
    for key in sorted(results):
        r = results[key]
        ok = r["status"] == r["expected"]
        flag = "PASS" if ok else "MISMATCH"
        line = f"  [{flag}] {key}: status={r['status']}, value={r['summary']}"
        if r["error"]:
            line += f", err={r['error'][:120]}"
        print(line)


def write_support_matrix_markdown(rows, out_md_path):
    lines = [
        "# Pulseq-zero adapter completeness matrix",
        "",
        "Built and probed with `pypulseq` and `pulseqzero` against the README",
        "section 4 coverage table. `pulseqzero` is invoked as a single drop-in",
        "module (no `pp_impl`, no `mr0_mode()`).",
        "",
        "api entry | pypulseq | pulseqzero | expected (pulseqzero)",
        "--- | --- | --- | ---",
    ]
    for key, pp_st, pz_st, expected in rows:
        mark = "" if pz_st == expected else " <-- MISMATCH"
        lines.append(
            f"{key} | {_STATUS_SYMBOL[pp_st]} | {_STATUS_SYMBOL[pz_st]}{mark} | {expected}"
        )
    lines.append("")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown report written: {out_md_path}")


def summarize_pass_fail(rows):
    pz_pass = sum(1 for _, _, pz, exp in rows if pz == exp)
    pz_total = len(rows)
    print(
        f"\npulseq-zero matches expected behavior on "
        f"{pz_pass}/{pz_total} probes."
    )


def main():
    pp_results, _ = probe_module(
        pypulseq, "pypulseq", "test_pypulseq_sequence_all_funcs.seq"
    )
    pz_results, _ = probe_module(
        pulseqzero, "pulseqzero", "test_pulseqzero_sequence_all_funcs.seq"
    )

    print_probe_details("pulseqzero", pz_results)
    rows = print_support_matrix(pp_results, pz_results)
    summarize_pass_fail(rows)
    write_support_matrix_markdown(
        rows, "test_pypulseq_pulseqzero_support_matrix.md"
    )


if __name__ == "__main__":
    main()
