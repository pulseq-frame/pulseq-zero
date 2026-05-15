import numpy as np
import importlib
import pypulseq as pp
import pulseqzero


def _maybe_call(func, *args, **kwargs):
    if callable(func):
        try:
            return True, func(*args, **kwargs), None
        except Exception as exc:  # pragma: no cover - smoke test error path
            return False, None, str(exc)
    return False, None, "not available in selected backend"


def _resolve_backend_func(pp_backend, backend: str, func_name: str):
    attr = getattr(pp_backend, func_name, None)
    if callable(attr):
        return attr

    # For plain pypulseq, some APIs are module-level (pp.func is a module).
    if backend == "pypulseq":
        try:
            mod = importlib.import_module(f"pypulseq.{func_name}")
            mod_attr = getattr(mod, func_name, None)
            if callable(mod_attr):
                return mod_attr
        except Exception:
            return None
        

    return None


def seq_func(
    out_seq_path: str = "test_pypulseq_sequence_all_funcs.seq",
    do_plot: bool = True,
    do_write: bool = True,
    backend: str = "pypulseq",
):
    """
    Build a broad pypulseq API test sequence.

    This intentionally exercises many common pypulseq 1.4.x APIs:
    - Opts, Sequence, set_definition
    - make_block_pulse, make_sinc_pulse(return_gz=True), make_delay
    - make_trapezoid (area / flat_area / amplitude styles)
    - make_extended_trapezoid, make_arbitrary_grad
    - make_adc, calc_duration
    - add_block, check_timing, calculate_kspace, plot, write
    """
    pp_backend = pp if backend == "pypulseq" else pulseqzero.pp_impl

    system = pp_backend.Opts(
        max_grad=28,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=20e-6,
        grad_raster_time=10e-6,
    )

    seq = pp_backend.Sequence(system=system)

    # Definitions that many QA tools expect.
    seq.set_definition("Name", "PyPulseq_AllFunctions_Test")
    seq.set_definition("FOV", [0.22, 0.22, 0.005])
    seq.set_definition("matrix", [64, 64, 1])
    seq.set_definition("TE", 0.03)
    seq.set_definition("TR", 0.2)

    # RF pulses
    rf_block = pp_backend.make_block_pulse(
        flip_angle=np.deg2rad(15),
        duration=1e-3,
        delay=system.rf_dead_time,
        system=system,
        use="excitation",
    )
    rf_sinc, gz, gzr = pp_backend.make_sinc_pulse(
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

    # Trapezoids in several creation styles
    gx_area = pp_backend.make_trapezoid(channel="x", area=4.0, duration=2e-3, system=system)
    gy_flat = pp_backend.make_trapezoid(channel="y", flat_area=8.0, flat_time=2e-3, system=system)
    gz_amp = pp_backend.make_trapezoid(channel="z", amplitude=6.0, duration=1.5e-3, system=system)

    # Extended trapezoid and arbitrary gradient
    ext_times = np.array([0, 0.5e-3, 1.2e-3, 2.0e-3])
    ext_amps = np.array([0.0, 0.8 * system.max_grad, -0.6 * system.max_grad, 0.0])
    gx_ext = pp_backend.make_extended_trapezoid(channel="x", times=ext_times, amplitudes=ext_amps, system=system)

    t = np.arange(0, 1e-3, system.grad_raster_time)
    arb_wave = 0.2 * system.max_grad * np.sin(2 * np.pi * t / t[-1])
    gy_arb = pp_backend.make_arbitrary_grad(channel="y", waveform=arb_wave, system=system)

    adc = pp_backend.make_adc(num_samples=100, dwell=20e-6, delay=20e-6, system=system)
    d_short = pp_backend.make_delay(500e-6)
    d_long = pp_backend.make_delay(2e-3)

    # Additional API smoke tests requested by user.
    gx2 = pp_backend.make_trapezoid(channel="x", area=2.0, duration=2e-3, system=system)
    gx_added = pp_backend.add_gradients([gx_area, gx2], system=system)

    aligned_events = pp_backend.align(right=[gx_area, gy_flat, gz_amp])

    k_for_ramps = np.vstack(
        [
            np.linspace(0.0, 1e-3, 8),
            np.zeros(8),
            np.zeros(8),
        ]
    )
    add_ramps_fn = _resolve_backend_func(pp_backend, backend, "add_ramps")
    ramps_ok, ramps_res, ramps_err = _maybe_call(add_ramps_fn, k_for_ramps, system=system)
    ramps_out = ramps_res if ramps_ok else []

    k0 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    k_end = np.array([[1e-3, 2e-3], [0.0, 0.0], [0.0, 0.0]])
    k_ramp, k_ramp_ok = pp_backend.calc_ramp(k0=k0, k_end=k_end, system=system)

    rf_bw = np.asarray(pp_backend.calc_rf_bandwidth(rf_sinc)).reshape(-1)[0]
    rf_center_t, rf_center_idx = pp_backend.calc_rf_center(rf_sinc)
    top_check_ok = False
    top_check_err = []
    top_check_duration = None

    compress_shape_fn = _resolve_backend_func(pp_backend, backend, "compress_shape")
    decompress_shape_fn = _resolve_backend_func(pp_backend, backend, "decompress_shape")
    convert_fn = _resolve_backend_func(pp_backend, backend, "convert")

    c_ok, compressed_arb, c_err = _maybe_call(compress_shape_fn, arb_wave)
    dc_ok, decompressed_arb, dc_err = _maybe_call(decompress_shape_fn, compressed_arb) if c_ok else (False, None, "compress_shape failed")
    conv_ok, converted_grad, conv_err = _maybe_call(convert_fn, 20.0, from_unit="mT/m", to_unit="Hz/m")

    gx_ext_area, gx_ext_area_times, gx_ext_area_amps = pp_backend.make_extended_trapezoid_area(
        area=1.0,
        channel="x",
        grad_start=0.0,
        grad_end=0.0,
        system=system,
    )
    interp_wave = pp_backend.points_to_waveform(
        amplitudes=np.array([0.0, 0.8 * system.max_grad, 0.0]),
        grad_raster_time=system.grad_raster_time,
        times=np.array([0.0, 0.5e-3, 1.0e-3]),
    )
    grad_from_traj, slew_from_traj = pp_backend.traj_to_grad(
        np.linspace(0.0, 1.0, 16), raster_time=system.grad_raster_time
    )

    rot_gx, rot_gy = pp_backend.rotate(gx_area, gy_flat, angle=np.pi / 8, axis="z")
    gx_scaled = pp_backend.scale_grad(gx_area, 0.5)
    gx_split_up, gx_split_flat, gx_split_down = pp_backend.split_gradient(gx_area, system=system)
    gx_split_1, gx_split_2 = pp_backend.split_gradient_at(gx_area, time_point=1e-3, system=system)

    rf_gauss, gz_gauss, gzr_gauss = pp_backend.make_gauss_pulse(
        flip_angle=np.deg2rad(30),
        duration=2e-3,
        slice_thickness=5e-3,
        apodization=0.25,
        time_bw_product=4,
        delay=system.rf_dead_time,
        return_gz=True,
        system=system,
        use="excitation",
    )
    rf_arb = pp_backend.make_arbitrary_rf(
        signal=np.hanning(128),
        flip_angle=np.deg2rad(20),
        delay=system.rf_dead_time,
        dwell=10e-6,
        system=system,
        use="excitation",
    )
    adc_label = pp_backend.make_label(label="LIN", type="SET", value=1)

    # Build sequence block-by-block (seq.add_block)
    seq.add_block(rf_block)
    seq.add_block(d_short)
    seq.add_block(rf_sinc, gz)
    seq.add_block(gzr)
    seq.add_block(gx_area, gy_flat)
    seq.add_block(gz_amp)
    seq.add_block(gx_ext)
    seq.add_block(gy_arb)
    seq.add_block(pp_backend.make_trapezoid(channel="x", flat_area=32.0, flat_time=2e-3, system=system), adc)
    seq.add_block(gx_added)
    seq.add_block(*aligned_events)
    seq.add_block(gx_ext_area)
    seq.add_block(rot_gx, rot_gy)
    seq.add_block(gx_scaled)
    seq.add_block(gx_split_up)
    seq.add_block(gx_split_flat)
    seq.add_block(gx_split_down)
    seq.add_block(gx_split_1)
    seq.add_block(gx_split_2)
    seq.add_block(rf_gauss, gz_gauss)
    seq.add_block(gzr_gauss)
    seq.add_block(rf_arb)
    seq.add_block(adc_label)
    seq.add_block(d_long)

    # Utility calls often used during development.
    ok, timing_report = seq.check_timing()
    top_check_fn = _resolve_backend_func(pp_backend, backend, "check_timing")
    top_ok, top_res, top_err = _maybe_call(top_check_fn, seq)
    if top_ok:
        top_check_ok, top_check_err = top_res
    else:
        top_check_ok, top_check_err = False, [top_err]
    ktraj_adc, ktraj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
    seq_duration_total, seq_duration_num_blocks, seq_duration_event_count = seq.duration()

    durations = {
        "rf_block": pp_backend.calc_duration(rf_block),
        "rf_sinc+gz": pp_backend.calc_duration(rf_sinc, gz),
        "gx_area": pp_backend.calc_duration(gx_area),
        "gy_flat": pp_backend.calc_duration(gy_flat),
        "gx_readout+adc": pp_backend.calc_duration(adc),
    }

    if do_plot:
        # time_range keeps plotting fast and avoids giant figures.
        seq.plot(time_range=[0, 0.03], plot_now=False)

    if do_write:
        seq.write(out_seq_path)

    return {
        "backend": backend,
        "sequence": seq,
        "timing_ok": ok,
        "timing_report": timing_report,
        "durations": durations,
        "api_smoke": {
            "add_block_ok": int(seq_duration_num_blocks) > 0,
            "make_trapezoid_ok": all(x is not None for x in [gx_area, gy_flat, gz_amp]),
            "make_block_pulse_ok": rf_block is not None,
            "make_sinc_pulse_ok": all(x is not None for x in [rf_sinc, gz, gzr]),
            "make_gauss_pulse_ok": all(x is not None for x in [rf_gauss, gz_gauss, gzr_gauss]),
            "make_arbitrary_grad_ok": gy_arb is not None,
            "add_ramps_shapes": [np.shape(x) for x in ramps_out],
            "add_ramps_error": ramps_err,
            "calc_ramp_ok": bool(k_ramp_ok),
            "calc_ramp_shape": np.shape(k_ramp),
            "calc_duration_rf_block": float(durations["rf_block"]),
            "calc_rf_bandwidth": float(rf_bw),
            "calc_rf_center": (float(rf_center_t), float(rf_center_idx)),
            "seq_duration_total": float(seq_duration_total),
            "seq_duration_num_blocks": int(seq_duration_num_blocks),
            "seq_duration_event_count_len": int(len(seq_duration_event_count)),
            "top_level_check_timing_ok": bool(top_check_ok),
            "top_level_check_timing_err": top_check_err,
            "top_level_check_timing_duration": top_check_duration,
            "compress_shape_samples": int(compressed_arb.num_samples) if c_ok else 0,
            "compress_shape_error": c_err,
            "decompress_shape_len": int(len(decompressed_arb)) if dc_ok else 0,
            "decompress_shape_error": dc_err,
            "convert_mTpm_to_Hzpm": float(converted_grad) if conv_ok else 0.0,
            "convert_error": conv_err,
            "make_extended_trapezoid_area_len": int(len(gx_ext_area_times)),
            "points_to_waveform_len": int(len(interp_wave)),
            "traj_to_grad_shape": np.shape(grad_from_traj),
            "traj_to_slew_shape": np.shape(slew_from_traj),
        },
        "ktraj_adc_shape": np.shape(ktraj_adc),
        "ktraj_shape": np.shape(ktraj),
        "t_excitation_shape": np.shape(t_excitation),
        "t_refocusing_shape": np.shape(t_refocusing),
        "t_adc_shape": np.shape(t_adc),
        "out_seq_path": out_seq_path if do_write else None,
    }


def print_smoke_results(result):
    print("Created sequence test.")
    print(f"Backend: {result['backend']}")
    print(f"Timing check: {'PASS' if result['timing_ok'] else 'FAIL'}")
    print(f"Wrote: {result['out_seq_path']}")
    print(f"k-space ADC shape: {result['ktraj_adc_shape']}")
    print("Smoke test results (SUCCESS/FAIL):")

    smoke_descriptions = {
        "add_ramps_shapes": "add_ramps generated ramped trajectories",
        "add_ramps_error": "add_ramps error detail (None means supported)",
        "calc_ramp_ok": "calc_ramp found a valid connection",
        "calc_ramp_shape": "calc_ramp returned trajectory shape",
        "calc_duration_rf_block": "calc_duration returned RF block duration",
        "calc_rf_bandwidth": "calc_rf_bandwidth returned positive bandwidth",
        "calc_rf_center": "calc_rf_center returned center time/index",
        "compress_shape_samples": "compress_shape produced sample metadata",
        "compress_shape_error": "compress_shape error detail (None means supported)",
        "convert_mTpm_to_Hzpm": "convert produced positive converted value",
        "convert_error": "convert error detail (None means supported)",
        "decompress_shape_len": "decompress_shape restored waveform length",
        "decompress_shape_error": "decompress_shape error detail (None means supported)",
        "make_extended_trapezoid_area_len": "make_extended_trapezoid_area returned times",
        "points_to_waveform_len": "points_to_waveform produced waveform",
        "seq_duration_total": "seq.duration returned total duration (raster units)",
        "seq_duration_num_blocks": "seq.duration returned number of blocks",
        "seq_duration_event_count_len": "seq.duration returned event-count array",
        "top_level_check_timing_duration": "top-level check_timing duration field",
        "top_level_check_timing_err": "top-level check_timing returned no errors",
        "top_level_check_timing_ok": "top-level check_timing passed",
        "traj_to_grad_shape": "traj_to_grad returned gradient samples",
        "traj_to_slew_shape": "traj_to_grad returned slew samples",
    }

    smoke_pass = {
        "add_ramps_shapes": len(result["api_smoke"]["add_ramps_shapes"]) > 0,
        "add_ramps_error": result["api_smoke"]["add_ramps_error"] is None,
        "calc_ramp_ok": bool(result["api_smoke"]["calc_ramp_ok"]),
        "calc_ramp_shape": len(result["api_smoke"]["calc_ramp_shape"]) > 0,
        "calc_duration_rf_block": result["api_smoke"]["calc_duration_rf_block"] > 0,
        "calc_rf_bandwidth": result["api_smoke"]["calc_rf_bandwidth"] > 0,
        "calc_rf_center": result["api_smoke"]["calc_rf_center"][0] >= 0,
        "compress_shape_samples": result["api_smoke"]["compress_shape_samples"] > 0,
        "compress_shape_error": result["api_smoke"]["compress_shape_error"] is None,
        "convert_mTpm_to_Hzpm": result["api_smoke"]["convert_mTpm_to_Hzpm"] > 0,
        "convert_error": result["api_smoke"]["convert_error"] is None,
        "decompress_shape_len": result["api_smoke"]["decompress_shape_len"] > 0,
        "decompress_shape_error": result["api_smoke"]["decompress_shape_error"] is None,
        "make_extended_trapezoid_area_len": result["api_smoke"]["make_extended_trapezoid_area_len"] > 0,
        "points_to_waveform_len": result["api_smoke"]["points_to_waveform_len"] > 0,
        "seq_duration_total": result["api_smoke"]["seq_duration_total"] > 0,
        "seq_duration_num_blocks": result["api_smoke"]["seq_duration_num_blocks"] > 0,
        "seq_duration_event_count_len": result["api_smoke"]["seq_duration_event_count_len"] > 0,
        "top_level_check_timing_duration": True,  # This field can be None in current pypulseq versions.
        "top_level_check_timing_err": len(result["api_smoke"]["top_level_check_timing_err"]) == 0,
        "top_level_check_timing_ok": bool(result["api_smoke"]["top_level_check_timing_ok"]),
        "traj_to_grad_shape": result["api_smoke"]["traj_to_grad_shape"][0] > 0,
        "traj_to_slew_shape": result["api_smoke"]["traj_to_slew_shape"][0] > 0,
    }

    for key in sorted(result["api_smoke"].keys()):
        status = "SUCCESS" if smoke_pass.get(key, False) else "FAIL"
        label = smoke_descriptions.get(key, key)
        value = result["api_smoke"][key]
        print(f"  - [{status}] {label}: {value}")
    if not result["timing_ok"]:
        print("Timing report:")
        for line in result["timing_report"]:
            print(f"  - {line}")


def run_mr0_mode_smoke_test(out_seq_path: str = "test_pulseqzero_mr0mode_sequence_all_funcs.seq"):
    mr0_smoke = {}
    result = None
    pp_backend = pulseqzero.pp_impl

    with pulseqzero.mr0_mode():
        seq_func_ok = True
        seq_func_err = None
        try:
            result = seq_func(
                out_seq_path=out_seq_path,
                do_plot=False,
                do_write=True,
                backend="pulseqzero",
            )
            seq = result["sequence"]
        except Exception as exc:
            # Full API smoke can fail in mr0_mode for backend-specific reasons.
            seq_func_ok = False
            seq_func_err = str(exc)
            system = pp_backend.Opts(
                max_grad=28,
                grad_unit="mT/m",
                max_slew=150,
                slew_unit="T/m/s",
                rf_ringdown_time=20e-6,
                rf_dead_time=100e-6,
                adc_dead_time=20e-6,
                grad_raster_time=10e-6,
            )
            seq = pp_backend.Sequence(system=system)
            seq.add_block(pp_backend.make_delay(1e-3))
            write_ok, _, _ = _maybe_call(
                seq.write,
                out_seq_path,
                create_signature=False,
                remove_duplicates=True,
            )
            result = {
                "backend": "pulseqzero (mr0_mode)",
                "sequence": seq,
                "timing_ok": True,
                "timing_report": [],
                "durations": {},
                "api_smoke": {},
                "ktraj_adc_shape": (),
                "out_seq_path": out_seq_path if write_ok else None,
            }

        # Smoke checks that are particularly relevant during optimization mode.
        duration_ok, duration_res, duration_err = _maybe_call(seq.duration)
        to_mr0_ok, to_mr0_res, to_mr0_err = _maybe_call(seq.to_mr0)

        # Exercise calc_duration in mr0_mode explicitly on a fresh delay event.
        d = pp_backend.make_delay(1e-3)
        calc_duration_ok, calc_duration_res, calc_duration_err = _maybe_call(pp_backend.calc_duration, d)

        # Explicit creator checks in mr0_mode, independent from full seq_func success.
        trap_ok, trap_res, trap_err = _maybe_call(
            pp_backend.make_trapezoid, channel="x", area=1.0, duration=1e-3, system=seq.system
        )
        block_ok, block_res, block_err = _maybe_call(
            pp_backend.make_block_pulse,
            flip_angle=np.deg2rad(10),
            duration=1e-3,
            delay=seq.system.rf_dead_time,
            system=seq.system,
            use="excitation",
        )
        sinc_ok, sinc_res, sinc_err = _maybe_call(
            pp_backend.make_sinc_pulse,
            flip_angle=np.deg2rad(20),
            duration=2e-3,
            slice_thickness=5e-3,
            apodization=0.5,
            time_bw_product=4,
            delay=seq.system.rf_dead_time,
            system=seq.system,
            return_gz=True,
            use="excitation",
        )
        gauss_ok, gauss_res, gauss_err = _maybe_call(
            pp_backend.make_gauss_pulse,
            flip_angle=np.deg2rad(20),
            duration=2e-3,
            slice_thickness=5e-3,
            apodization=0.5,
            time_bw_product=4,
            delay=seq.system.rf_dead_time,
            return_gz=True,
            system=seq.system,
            use="excitation",
        )
        arb_ok, arb_res, arb_err = _maybe_call(
            pp_backend.make_arbitrary_grad,
            channel="y",
            waveform=np.zeros(16),
            system=seq.system,
        )
        seq_add_test = pp_backend.Sequence(system=seq.system)
        add_block_ok, _, add_block_err = _maybe_call(seq_add_test.add_block, pp_backend.make_delay(1e-3))

        mr0_smoke = {
            "seq_func_ok": bool(seq_func_ok),
            "seq_func_error": seq_func_err,
            "add_block_ok": bool(add_block_ok),
            "add_block_error": add_block_err,
            "make_trapezoid_ok": bool(trap_ok),
            "make_trapezoid_error": trap_err,
            "make_block_pulse_ok": bool(block_ok),
            "make_block_pulse_error": block_err,
            "make_sinc_pulse_ok": bool(sinc_ok),
            "make_sinc_pulse_error": sinc_err,
            "make_gauss_pulse_ok": bool(gauss_ok),
            "make_gauss_pulse_error": gauss_err,
            "make_arbitrary_grad_ok": bool(arb_ok),
            "make_arbitrary_grad_error": arb_err,
            "duration_ok": bool(duration_ok),
            "duration_error": duration_err,
            "duration_total": float(duration_res[0]) if duration_ok else 0.0,
            "duration_num_blocks": int(duration_res[1]) if duration_ok else 0,
            "to_mr0_ok": bool(to_mr0_ok),
            "to_mr0_error": to_mr0_err,
            "to_mr0_type": str(type(to_mr0_res)) if to_mr0_ok else None,
            "calc_duration_ok": bool(calc_duration_ok),
            "calc_duration_error": calc_duration_err,
            "calc_duration_value": float(calc_duration_res) if calc_duration_ok else 0.0,
        }

    result["mr0_mode_smoke"] = mr0_smoke
    result["backend"] = "pulseqzero (mr0_mode)"
    return result


def print_mr0_mode_smoke_results(result):
    print("MR0 mode smoke results (SUCCESS/FAIL):")
    checks = result["mr0_mode_smoke"]
    pass_map = {
        "seq_func_ok": checks["seq_func_ok"],
        "seq_func_error": checks["seq_func_error"] is None,
        "duration_ok": checks["duration_ok"],
        "duration_error": checks["duration_error"] is None,
        "duration_total": checks["duration_total"] > 0,
        "duration_num_blocks": checks["duration_num_blocks"] > 0,
        "to_mr0_ok": checks["to_mr0_ok"],
        "to_mr0_error": checks["to_mr0_error"] is None,
        "to_mr0_type": checks["to_mr0_type"] is not None,
        "calc_duration_ok": checks["calc_duration_ok"],
        "calc_duration_error": checks["calc_duration_error"] is None,
        "calc_duration_value": checks["calc_duration_value"] > 0,
    }
    for key in [
        "seq_func_ok",
        "seq_func_error",
        "duration_ok",
        "duration_error",
        "duration_total",
        "duration_num_blocks",
        "to_mr0_ok",
        "to_mr0_error",
        "to_mr0_type",
        "calc_duration_ok",
        "calc_duration_error",
        "calc_duration_value",
    ]:
        status = "SUCCESS" if pass_map.get(key, False) else "FAIL"
        print(f"  - [{status}] {key}: {checks[key]}")


def _support_matrix_from_results(result_pypulseq, result_pulseqzero, result_mr0):
    p = result_pypulseq["api_smoke"]
    z = result_pulseqzero["api_smoke"]
    m = result_mr0["mr0_mode_smoke"]

    matrix = {
        "add_block": (p["add_block_ok"], z["add_block_ok"], m["add_block_ok"]),
        "add_ramps": (p["add_ramps_error"] is None, z["add_ramps_error"] is None, False),
        "make_trapezoid": (p["make_trapezoid_ok"], z["make_trapezoid_ok"], m["make_trapezoid_ok"]),
        "make_block_pulse": (p["make_block_pulse_ok"], z["make_block_pulse_ok"], m["make_block_pulse_ok"]),
        "make_sinc_pulse": (p["make_sinc_pulse_ok"], z["make_sinc_pulse_ok"], m["make_sinc_pulse_ok"]),
        "make_gauss_pulse": (p["make_gauss_pulse_ok"], z["make_gauss_pulse_ok"], m["make_gauss_pulse_ok"]),
        "make_arbitrary_grad": (p["make_arbitrary_grad_ok"], z["make_arbitrary_grad_ok"], m["make_arbitrary_grad_ok"]),
        "calc_ramp": (p["calc_ramp_ok"], z["calc_ramp_ok"], False),
        "calc_duration": (p["calc_duration_rf_block"] > 0, z["calc_duration_rf_block"] > 0, m["calc_duration_ok"]),
        "calc_rf_bandwidth": (p["calc_rf_bandwidth"] > 0, z["calc_rf_bandwidth"] > 0, False),
        "calc_rf_center": (p["calc_rf_center"][0] >= 0, z["calc_rf_center"][0] >= 0, False),
        "check_timing_top_level": (p["top_level_check_timing_ok"], z["top_level_check_timing_ok"], False),
        "compress_shape": (p["compress_shape_error"] is None, z["compress_shape_error"] is None, False),
        "decompress_shape": (p["decompress_shape_error"] is None, z["decompress_shape_error"] is None, False),
        "convert": (p["convert_error"] is None, z["convert_error"] is None, False),
        "make_extended_trapezoid_area": (p["make_extended_trapezoid_area_len"] > 0, z["make_extended_trapezoid_area_len"] > 0, False),
        "points_to_waveform": (p["points_to_waveform_len"] > 0, z["points_to_waveform_len"] > 0, False),
        "traj_to_grad": (p["traj_to_grad_shape"][0] > 0, z["traj_to_grad_shape"][0] > 0, False),
        "traj_to_slew": (p["traj_to_slew_shape"][0] > 0, z["traj_to_slew_shape"][0] > 0, False),
        "seq.duration": (p["seq_duration_total"] > 0, z["seq_duration_total"] > 0, m["duration_ok"]),
        "seq.to_mr0": (False, False, m["to_mr0_ok"]),
    }
    return matrix


def _status(v: bool) -> str:
    return "SUCCESS" if v else "FAIL"


def _status_md(v: bool) -> str:
    return "✅" if v else "❌"


def print_support_matrix(result_pypulseq, result_pulseqzero, result_mr0):
    matrix = _support_matrix_from_results(result_pypulseq, result_pulseqzero, result_mr0)
    print("")
    print("func_name | pypulseq | pulseqzero | pulseqzero.mr0mode")
    print("--- | --- | --- | ---")
    for fn in sorted(matrix.keys()):
        a, b, c = matrix[fn]
        print(f"{fn} | {_status(a)} | {_status(b)} | {_status(c)}")
    return matrix


def write_support_matrix_markdown(
    matrix: dict, out_md_path: str = "test_pypulseq_pulseqzero_support_matrix.md"
):
    lines = [
        "# PyPulseq vs PulseqZero Support Matrix",
        "",
        "func_name | pypulseq | pulseqzero | pulseqzero.mr0mode",
        "--- | --- | --- | ---",
    ]
    for fn in sorted(matrix.keys()):
        a, b, c = matrix[fn]
        lines.append(f"{fn} | {_status_md(a)} | {_status_md(b)} | {_status_md(c)}")
    lines.append("")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown report written: {out_md_path}")


if __name__ == "__main__":
    result_pypulseq = seq_func(
        out_seq_path="test_pypulseq_sequence_all_funcs.seq",
        do_plot=False,
        do_write=True,
        backend="pypulseq",
    )
    # Keep detailed sections available, but final summary below is the main output.
    result_pulseqzero = seq_func(
        out_seq_path="test_pulseqzero_sequence_all_funcs.seq",
        do_plot=False,
        do_write=True,
        backend="pulseqzero",
    )
    result_pulseqzero_mr0_mode = run_mr0_mode_smoke_test()

    matrix = print_support_matrix(result_pypulseq, result_pulseqzero, result_pulseqzero_mr0_mode)
    write_support_matrix_markdown(matrix)
