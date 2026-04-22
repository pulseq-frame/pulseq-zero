# UNIFY: removing the `mr0_mode` distinction

## Goal

Make pulseq-zero a single, always-on facade. User scripts should read:

```python
import pulseqzero as pp

seq = pp.Sequence()
rf, gz, _ = pp.make_sinc_pulse(...)
seq.add_block(rf, gz)

mr0_seq = seq.to_mr0()     # for simulation / optimization
seq.write("scan.seq")       # for the scanner
```

No `pp = pulseqzero.pp_impl`, no `with pulseqzero.mr0_mode():`, no mode flag, no `is_mr0_mode()`. The adapter *is* the library. `to_mr0()` stays. `.seq` export happens by translating the adapter's in-memory sequence into PyPulseq calls at write time.

The north star: **pulseq-zero is a drop-in replacement for PyPulseq.** Any existing PyPulseq script should run under pulseq-zero by swapping `import pypulseq as pp` for `import pulseqzero as pp` — nothing else. On top of that, `seq.to_mr0()` gives you a differentiable sequence for free. Breakage against earlier pulseq-zero versions is acceptable if it buys a cleaner surface; there is no deprecation shim.

The work lands as a **single PR**: one breaking commit that unifies the API, delegates pulse-shape generation to PyPulseq, and wires `.seq` export end-to-end. No intermediate half-states.

## Current state (for context)

Today the facade works by **mode-swapping**:

- [src/pulseqzero/\_\_init\_\_.py](src/pulseqzero/__init__.py) defines `Impl`, a singleton whose attributes (`Sequence`, `make_sinc_pulse`, …) are rebound by either `use_pypulseq()` or `use_pulseqzero()`.
- `pp_impl = Impl()` starts in PyPulseq mode. Every attribute is a `torch_to_numpy`-wrapped forwarder to real PyPulseq.
- `with mr0_mode():` flips attributes to the adapter ([src/pulseqzero/adapter/](src/pulseqzero/adapter/)). Inside the block, the same call names build differentiable torch dataclasses instead.
- `seq.to_mr0()` only exists in mr0 mode, on the adapter's `Sequence`.
- `.seq` export only works in PyPulseq mode (the adapter's `Sequence.write()` just warns).

The two "modes" are really **two evaluators of the same script**.

## Proposed architecture

**One evaluator (the adapter). PyPulseq is used only at export, and — under the hood — for pulse-shape generation.**

```
 user script
    │
    ▼                           ┌────────────────────┐
 pulseqzero.make_sinc_pulse ─►  │ adapter dataclass  │
 pulseqzero.make_trapezoid ─►   │ graph inside a     │
 pulseqzero.Sequence.add_block  │ pp.Sequence        │
                                └─────────┬──────────┘
                                          │
                        ┌─────────────────┴────────────────┐
                        ▼                                   ▼
                  seq.to_mr0()                        seq.write(path)
                       │                                    │
                       ▼                                    ▼
              MRzeroCore.Sequence               PyPulseq calls, .seq file
              (already implemented)             (new translator)
```

The adapter dataclasses (`Pulse`, `TrapGrad`, `FreeGrad`, `Adc`, `Delay`) already carry every number PyPulseq needs to reconstruct the event. The `.seq` export is therefore a **walk-and-translate**, not a replay.

### Why this over "record the calls and replay to PyPulseq"

The `CLAUDE.md`-shaped alternative ("track all call names and kwargs, then call PyPulseq with the recorded args") sounds clean but has two problems:

1. **Attribute reads leak through.** User code does `pp.make_trapezoid(amplitude=gz.amplitude, ...)`. Whatever `gz` is, its `.amplitude` has to be a usable number *right now*, not a deferred symbol. So something concrete has to be evaluated eagerly — which means you're already running the adapter. Adding a recording tape on top of it is duplicated bookkeeping.
2. **`to_mr0()` needs the adapter anyway.** If we're already building the differentiable graph to support simulation, doing a second pass through PyPulseq at export time is all we're missing.

So: **adapter is the single source of truth; export translates.**

### Pulse-shape generation: delegate to PyPulseq

The adapter currently hand-rolls the pulse envelopes in the `generate_shape()` closures of `make_sinc_pulse` / `make_gauss_pulse` / `make_block_pulse` (see [src/pulseqzero/adapter/pulses.py](src/pulseqzero/adapter/pulses.py)). It also hand-rolls the companion `gz` / `gzr` trapezoids for the `return_gz=True` paths. That is duplicated PyPulseq logic, and the more we duplicate the more likely we drift (and the more we break when PyPulseq 1.6 ships).

Preferred approach: **call PyPulseq's own factory once at construction time, take its shape, and wrap it in an adapter `Pulse`.**

```python
# inside pulseqzero.make_sinc_pulse (sketch)
def make_sinc_pulse(flip_angle, *, system=None, duration=4e-3, ..., return_gz=False, **kw):
    import pypulseq
    fa_np = _n(flip_angle)   # detached numpy scalar
    result = pypulseq.make_sinc_pulse(
        flip_angle=fa_np, system=system or Opts.default,
        duration=float(duration), ..., return_gz=return_gz,
    )
    pp_rf, *pp_gz = result if isinstance(result, tuple) else (result,)
    rf = Pulse(
        flip_angle=flip_angle,             # live torch tensor
        shape_dur=pp_rf.shape_dur,
        delay=pp_rf.delay,
        freq_offset=pp_rf.freq_offset,
        phase_offset=pp_rf.phase_offset,
        ringdown_time=pp_rf.ringdown_time,
        shim_array=pp_rf.shim_array,
        shape=(pp_rf.t, pp_rf.signal),     # numpy, ready to use
        use=pp_rf.use,
        _pp_factory="make_sinc_pulse",
        _pp_kwargs={"duration": float(duration), "time_bw_product": ..., ...},
    )
    if return_gz:
        gz, gzr = pp_gz
        return rf, _wrap_trap(gz), _wrap_trap(gzr)
    return rf
```

**Trade-offs:**

- **Autograd through pulse shape is lost.** Duration-as-shape-input, TBW, apodization, slice_thickness, etc. become non-differentiable because they feed into PyPulseq as numpy. `flip_angle` / `phase_offset` / `freq_offset` / `delay` autograd is unaffected — we keep the live tensors separately on `Pulse`, and `seq_convert.integrate_pulse` already reconnects `flip_angle` via the `window_area / full_area` ratio trick.
- **Code deletion.** Three `generate_shape` closures, the `return_gz` / `return_gzr` math, and the flip-angle detach dance all disappear. The adapter's pulse layer becomes a thin wrapper.
- **Shape provenance for `.seq` export.** Stashing `_pp_factory` / `_pp_kwargs` means we can recall the exact same factory at write time and get a pretty `.seq` (a native sinc block instead of arbitrary waveform). Fallback is always `make_arbitrary_rf(signal=ev.shape[1], ...)`. `_pp_kwargs` only contains the shape-defining fields — mutable fields like `flip_angle` / `phase_offset` / `freq_offset` / `delay` are re-read from the live `Pulse` at write time so post-construction user edits are honored (e.g. `rf.phase_offset = new_phase`).
- **Dependency cost.** PyPulseq must be importable at adapter import time. It already needs to be in the env; today we import it only on the `use_pypulseq()` path. Promoting that to unconditional import is a non-issue in practice.
- **Speed.** Calling PyPulseq per pulse is similar cost to today's numpy closure firing on `to_mr0()`. No regression expected; caching across identical kwarg calls is possible if it ever shows up.

Bottom line: **delegate shape generation to PyPulseq; lose pulse-shape autograd; gain ~150 lines of deleted duplicated logic and automatic 1.4-vs-1.5 compatibility.** Autograd on `flip_angle` — the only pulse parameter anyone optimizes in practice — keeps working.

If pulse-shape autograd is ever genuinely needed, the closure approach can be added back as an opt-in `differentiable=True` flag on a per-factory basis. Unlikely to matter.

## Differentiability guarantees

The unification intentionally preserves autograd for the parameters anyone actually optimizes. Explicitly:

**Differentiable (must stay differentiable through `to_mr0()`):**
- RF `flip_angle`, `phase_offset`, `freq_offset`
- ADC `phase_offset`, `freq_offset`
- Gradient `amplitude` (trapezoid and arbitrary)
- All timings: `delay`, `duration`, `rise_time`, `flat_time`, `fall_time`, `dwell`

**Not differentiable (acceptable, listed under "Post-unification follow-ups" at the bottom):**
- RF pulse shape samples and the parameters that only affect them (duration-when-used-for-shape, `time_bw_product`, `apodization`, `center_pos`, `slice_thickness` as it feeds shape generation)
- Gradient waveform samples for arbitrary gradients (only the amplitude scale is differentiable)
- `Opts` fields (max_grad, rasters, dead times, etc.) — intentionally numeric

If pulse-shape autograd is ever needed later, it comes back via an opt-in flag on the factory; not in scope now.

## User-visible API after unification

| before                                               | after                          |
| ---------------------------------------------------- | ------------------------------ |
| `import pulseqzero`                                  | `import pulseqzero as pp`      |
| `pp = pulseqzero.pp_impl`                            | *(removed — will raise `AttributeError`)* |
| `with pulseqzero.mr0_mode(): seq = build()`          | `seq = build()`                |
| `seq.to_mr0()` *(only in mr0 block)*                 | `seq.to_mr0()` *(always)*      |
| `seq.write(path)` *(only outside mr0 block)*         | `seq.write(path)` *(always)*   |
| `if pulseqzero.is_mr0_mode(): ...`                   | *(removed — users shouldn't branch; unsupported calls raise `NotImplementedError`)* |

Inside the script the `pp.X(...)` call surface is unchanged. [demo/write_tse.py](demo/write_tse.py) becomes a one-line diff: swap `import pulseqzero; pp = pulseqzero.pp_impl` for `import pulseqzero as pp`. Nothing else.

This is a **breaking change** against earlier pulseq-zero releases. Lands as part of the 1.0 release off the current branch (0.3 → 1.0 on PyPI) with a release note calling out the migration (two mechanical edits per script).

## Concrete change list

### 1. [src/pulseqzero/\_\_init\_\_.py](src/pulseqzero/__init__.py) — rewrite

Delete `Impl`, `pp_impl`, `use_pypulseq`, `use_pulseqzero`, `mr0_mode`, `is_mr0_mode`, `torch_to_numpy`, `convert_tensor`. Replace with direct re-exports from the adapter:

```python
from .math import ceil, floor, round

from .adapter import (
    Opts,
    Sequence,
    calc_duration, calc_rf_bandwidth, calc_rf_center, calc_SAR,
    make_adc, make_delay, make_trigger, make_digital_output_pulse,
    make_label, get_supported_labels,
    make_trapezoid, make_arbitrary_grad, make_extended_trapezoid,
    make_extended_trapezoid_area, scale_grad, split_gradient, split_gradient_at,
    add_gradients,
    make_sinc_pulse, make_gauss_pulse, make_block_pulse, make_arbitrary_rf,
)
```

File shrinks from ~155 lines to ~25.

### 2. [src/pulseqzero/adapter/opts.py](src/pulseqzero/adapter/opts.py) — re-export `pypulseq.Opts`

The adapter's `Opts` is not differentiable and never needed to be a torch-aware wrapper. Drop the custom dataclass, re-export PyPulseq's directly:

```python
# adapter/opts.py
from pypulseq import Opts
Opts.default = Opts()    # keep the module-level singleton the adapter already relies on
```

This removes one file's worth of maintenance and guarantees 1:1 field parity with whichever PyPulseq version is installed (`B0`, `adc_samples_limit`, etc. come along for free). Audit callers inside the adapter for attribute names that might have been renamed; expect a small number of `rf_dead_time` / `rf_raster_time` reads that Just Work.

### 3. [src/pulseqzero/adapter/pulses.py](src/pulseqzero/adapter/pulses.py) — delegate shape generation

Replace each `generate_shape` closure with a one-line call to the corresponding PyPulseq factory (see sketch above). Store the returned numpy `(t, signal)` pair on the `Pulse` as `shape`. Stash `_pp_factory` / `_pp_kwargs` — containing only **shape-defining** fields (duration, TBW, apodization, center_pos, slice_thickness…) — for later `.seq` re-construction. Same pattern for `make_gauss_pulse`, `make_block_pulse`, `make_arbitrary_rf`.

Also delete the hand-rolled `gz` / `gzr` trapezoid construction in the `return_gz` branches — PyPulseq returns those; wrap them in the adapter `TrapGrad`.

### 4. [src/pulseqzero/adapter/seq_convert.py](src/pulseqzero/adapter/seq_convert.py) — read from stashed shape

`integrate_pulse` currently calls `rf._generate_shape()`. After §3, it reads the already-materialized `rf.shape` tuple. The `window_area / full_area` area-ratio trick that reconnects `flip_angle` autograd is untouched.

### 5. [src/pulseqzero/adapter/sequence.py](src/pulseqzero/adapter/sequence.py) — implement `write()` and `to_pypulseq()`

`write()` is currently a stub that warns. Also fix the existing signature bug at [sequence.py:197](src/pulseqzero/adapter/sequence.py#L197) — no defaults for `create_signature` / `remove_duplicates`.

New body walks `self.blocks`, translates each event to a PyPulseq event, adds it to a `pypulseq.Sequence`, and calls its `.write()`. Translation is **lazy** (no caching) and emits a `warnings.warn` on every materialization so users notice when it fires inside an optimization loop:

```python
def write(self, name, create_signature=True, remove_duplicates=True):
    return self.to_pypulseq().write(
        name, create_signature=create_signature, remove_duplicates=remove_duplicates,
    )

def to_pypulseq(self):
    import pypulseq, warnings
    warnings.warn(
        "pulseqzero.Sequence: translating to PyPulseq — expect a delay for large sequences.",
        stacklevel=2,
    )
    pp_seq = pypulseq.Sequence(system=self.system)    # self.system is pypulseq.Opts (see §2)
    for block in self.blocks:
        pp_seq.add_block(*[_event_to_pp(ev) for ev in block])
    for k, v in self.definitions.items():
        pp_seq.set_definition(key=k, value=_n(v))
    return pp_seq
```

If the warning starts firing inside tight loops, add a translated-PyPulseq-seq cache on the adapter `Sequence` with invalidation on `add_block` / `set_definition` as a follow-up.

### 6. New file: `src/pulseqzero/adapter/to_pypulseq.py`

Hosts the per-event translators and the `_n(x)` helper (detach-and-to-numpy, moved from today's `torch_to_numpy`).

```python
def _event_to_pp(ev):
    if isinstance(ev, Delay):      return pypulseq.make_delay(_n(ev.delay))
    if isinstance(ev, Adc):        return pypulseq.make_adc(
        num_samples=int(ev.num_samples), dwell=_n(ev.dwell),
        delay=_n(ev.delay), freq_offset=_n(ev.freq_offset),
        phase_offset=_n(ev.phase_offset),
    )
    if isinstance(ev, TrapGrad):   return pypulseq.make_trapezoid(
        channel=ev.channel, amplitude=_n(ev.amplitude),
        rise_time=_n(ev.rise_time), flat_time=_n(ev.flat_time),
        fall_time=_n(ev.fall_time), delay=_n(ev.delay),
    )
    if isinstance(ev, FreeGrad):   return pypulseq.make_arbitrary_grad(
        channel=ev.channel, waveform=_n(ev.waveform),
        delay=_n(ev.delay), first=_n(ev.first), last=_n(ev.last),
    )
    if isinstance(ev, Pulse):
        if ev._pp_factory is not None:
            # Re-call the factory for a pretty .seq.
            # Override mutable fields from the live Pulse — the user may have
            # mutated rf.flip_angle / rf.phase_offset / rf.delay after
            # construction, so the stored snapshot alone would be wrong.
            fac = getattr(pypulseq, ev._pp_factory)
            kwargs = {k: _n(v) for k, v in ev._pp_kwargs.items()}
            kwargs["flip_angle"]   = _n(ev.flip_angle)
            kwargs["phase_offset"] = _n(ev.phase_offset)
            kwargs["freq_offset"]  = _n(ev.freq_offset)
            kwargs["delay"]        = _n(ev.delay)
            if "use" in inspect.signature(fac).parameters:
                kwargs["use"] = ev.use
            result = fac(**kwargs)
            return result[0] if isinstance(result, tuple) else result
        # Fallback: arbitrary waveform from the stored shape.
        _, signal = ev.shape
        return pypulseq.make_arbitrary_rf(
            signal=signal, flip_angle=_n(ev.flip_angle),
            freq_offset=_n(ev.freq_offset),
            phase_offset=_n(ev.phase_offset),
            delay=_n(ev.delay), use=ev.use,
        )
    raise TypeError(f"Unknown event type: {type(ev).__name__}")
```

`_n(x)` is `x.detach().cpu().numpy()` if torch, else `np.asarray(x)` — identical to today's `convert_tensor`, just moved into the adapter package.

### 7. Unsupported PyPulseq features — implement or raise, don't ask the user to branch

Two rules, no third:

1. **Functions we can forward: implement them.** `Sequence.calculate_pns`, `Sequence.test_report`, `Sequence.plot`, `Sequence.paper_plot`, `Sequence.check_timing`, `calc_rf_bandwidth`, `calc_rf_center`, etc. all internally call `self.to_pypulseq()` (or the per-event translator) and forward. User scripts never branch.

   ```python
   class Sequence:
       def calculate_pns(self, *args, **kwargs):
           return self.to_pypulseq().calculate_pns(*args, **kwargs)
   ```

2. **Functions we genuinely cannot support: raise a clear `NotImplementedError`.** Candidates: sigpy-based pulse designers with no sigpy installed, adiabatic pulses whose differentiable reimplementation we haven't done, PyPulseq-1.5-only features we haven't wired yet. Message must name the function and say *why* it isn't supported and what the workaround is.

   ```python
   def make_adiabatic_pulse(*args, **kwargs):
       raise NotImplementedError(
           "pulseqzero.make_adiabatic_pulse is not implemented. "
           "Workaround: build the pulse with pypulseq directly, then wrap "
           "the resulting signal via pulseqzero.make_arbitrary_rf(signal=..., ...)."
       )
   ```

`seq.to_pypulseq()` is the single explicit escape hatch: anyone who wants the native PyPulseq object (to call an exotic PyPulseq method, hand off to a third-party tool, etc.) gets one.

### 8. Callsite sweeps

All user scripts migrate to the new import. There is no shim.

- [demo/write_tse.py](demo/write_tse.py): swap the two-line import for `import pulseqzero as pp`. Drop the running changelog header's "swapped `import pypulseq as pp` for the pulseq-zero facade" bullet.
- [demo/main.py](demo/main.py): drop `with pulseqzero.mr0_mode():` from `simulate()`; call `build_tse(...)` directly.
- [README.md](README.md) §2 and §4: rewrite the "Usage" and "Additional API" sections to drop all mode talk. Keep (and update) the coverage table.

## Acceptance tests

Both must pass before merging.

1. **Optimization still converges.** `uv run demo/main.py` completes 30 Adam iterations with monotonically-decreasing SAR and non-NaN data loss, identical behavior to pre-unification. Proves `flip_angle` autograd still threads from the adapter through `to_mr0()`.
2. **`.seq` export is byte-identical to the PyPulseq reference.** Run [demo/write_tse.py](demo/write_tse.py) twice:
   - once after changing its `import pulseqzero as pp` to `import pypulseq as pp` (PyPulseq reference),
   - once with the unified `import pulseqzero as pp` path that goes through the adapter + `Sequence.write()`.

   Diff the two `.seq` files. They must match byte-for-byte on the event stream (ignoring timestamp / signature lines that PyPulseq always generates afresh). This is the single acceptance test that guarantees the translator round-trips: same script, same inputs, same output, regardless of which backend actually emitted the file.

## Effort estimate

| area                                                     | LoC delta      | risk                      |
| -------------------------------------------------------- | -------------- | ------------------------- |
| `__init__.py` rewrite                                    | −130 / +25     | low                       |
| `adapter/opts.py` → re-export pypulseq.Opts              | −30 / +5       | low (audit callers)       |
| `adapter/pulses.py` delegate to PyPulseq                 | −120 / +60     | medium (shape fidelity)   |
| `adapter/sequence.py` `write()` + `to_pypulseq()`        | +60            | medium (export fidelity)  |
| `adapter/to_pypulseq.py` (new)                           | +100           | medium                    |
| Forwarding stubs for unsupported-but-implementable fns   | +60            | low                       |
| Clear-error raisers for truly-unsupported fns            | +20            | low                       |
| Callsite sweep ([demo/](demo/))                          | ~±10           | trivial                   |
| README rewrite of §2 and §4                              | prose          | trivial                   |

All in: on the order of **250 lines of net code change and a README rewrite**. The hard parts are (a) the per-event PyPulseq translators and (b) getting the `.seq` output to round-trip byte-identically against the PyPulseq reference — that is exactly what the acceptance test above checks.

## Implementation TODO

Execute in this order within a single PR. Each item is a concrete code change; check off as you go.

### Core API rewrite

- [x] Delete `Impl`, `pp_impl`, `use_pypulseq`, `use_pulseqzero`, `mr0_mode`, `is_mr0_mode`, `torch_to_numpy`, `convert_tensor` from [src/pulseqzero/\_\_init\_\_.py](src/pulseqzero/__init__.py).
- [x] Replace with adapter re-exports (see §1 snippet).
- [x] Replace [src/pulseqzero/adapter/opts.py](src/pulseqzero/adapter/opts.py) body with `from pypulseq import Opts; Opts.default = Opts()`.
- [x] Grep the adapter for callers of the old `Opts` dataclass — fix any attribute names that differ from `pypulseq.Opts`. (All attributes — `max_grad`, `max_slew`, `grad_raster_time`, `rf_raster_time`, `rf_dead_time`, `rf_ringdown_time`, `adc_dead_time`, `adc_raster_time`, `gamma`, `B0` — are 1:1 with `pypulseq.Opts`.)
- [ ] Move `_n(x)` (torch-tensor → numpy helper, formerly `convert_tensor`) into the adapter package. *(Deferred until the `to_pypulseq` translator lands in §6.)*

### Pulse-shape delegation

- [x] Rewrite `make_sinc_pulse` in [src/pulseqzero/adapter/pulses.py](src/pulseqzero/adapter/pulses.py) to call `pypulseq.make_sinc_pulse` and wrap the returned objects.
- [x] Same for `make_gauss_pulse`, `make_block_pulse`, `make_arbitrary_rf`.
- [x] Add `shape: tuple[np.ndarray, np.ndarray]`, `_pp_factory: str`, `_pp_kwargs: dict` fields to the `Pulse` dataclass. Drop the `_generate_shape` closure.
- [x] Ensure `_pp_kwargs` holds **only shape-defining fields** (duration, TBW, apodization, center_pos, slice_thickness, …). Mutable fields (`flip_angle`, `phase_offset`, `freq_offset`, `delay`) stay on the `Pulse` as live tensors.
- [x] Delete the `return_gz` / `return_gzr` hand-rolled trapezoid math; wrap the `gz`/`gzr` that PyPulseq returns into adapter `TrapGrad`.
- [x] Update [src/pulseqzero/adapter/seq_convert.py](src/pulseqzero/adapter/seq_convert.py) `integrate_pulse` to read `rf.shape` instead of calling `rf._generate_shape()`. Keep the `window_area / full_area` area-ratio autograd reconnect. Plot path in [sequence.py](src/pulseqzero/adapter/sequence.py) also reads `event.shape`.

### `.seq` export

- [ ] New file [src/pulseqzero/adapter/to_pypulseq.py](src/pulseqzero/adapter/to_pypulseq.py) hosting `_event_to_pp(ev)` for every adapter event type (`Pulse`, `TrapGrad`, `FreeGrad`, `Adc`, `Delay`).
- [ ] Pulse translator must re-call `_pp_factory` with `_pp_kwargs` **and override `flip_angle` / `phase_offset` / `freq_offset` / `delay` from the live `Pulse`** so post-construction user edits are honored.
- [ ] Pulse translator fallback for `Pulse` objects without `_pp_factory` (e.g. `make_arbitrary_rf` callers): emit `pypulseq.make_arbitrary_rf(signal=…)` from the stored shape.
- [ ] Implement `Sequence.to_pypulseq()` on the adapter — lazy (no cache), emit `warnings.warn(...)` on every call so hot-loop usage is visible.
- [ ] Implement `Sequence.write(name, create_signature=True, remove_duplicates=True)` as a thin wrapper over `self.to_pypulseq().write(...)`. This also fixes the existing signature bug at [sequence.py:197](src/pulseqzero/adapter/sequence.py#L197) (missing defaults).

### Unsupported features

- [ ] Wire forwarders for everything that can be delegated: `Sequence.calculate_pns`, `Sequence.test_report`, `Sequence.plot`, `Sequence.paper_plot`, `Sequence.check_timing`, `calc_rf_bandwidth`, `calc_rf_center`, etc. Each body is `return self.to_pypulseq().foo(*args, **kwargs)`.
- [ ] For things we genuinely can't do (sigpy pulses, adiabatic pulses, any 1.5-only features we haven't wired): `raise NotImplementedError` with a message that names the function, says why, and points to the workaround.

### Callsite sweep

- [ ] [demo/write_tse.py](demo/write_tse.py): replace `import pulseqzero; pp = pulseqzero.pp_impl` with `import pulseqzero as pp`. Drop the "swapped import" bullet from its running changelog.
- [ ] [demo/main.py](demo/main.py): drop `with pulseqzero.mr0_mode():` from `simulate()`; call `build_tse(...)` and `.to_mr0()` directly.
- [ ] [README.md](README.md) §2 and §4: drop all mode talk; update the coverage table.

### Acceptance tests

- [ ] `uv run demo/main.py` runs end-to-end, completes 30 Adam iterations, SAR decreases monotonically, data loss is non-NaN. (Identical behavior to pre-unification.)
- [ ] Clone [demo/write_tse.py](demo/write_tse.py) into a one-off script that can be run twice: once with `import pulseqzero as pp`, once with `import pypulseq as pp`. Run both, save `.seq` outputs, diff them. Must match byte-for-byte (ignoring timestamp / signature lines).

### Release

- [ ] Confirm version is 1.0.0 in [pyproject.toml](pyproject.toml) (already bumped on this branch).
- [ ] [CHANGELOG.md](CHANGELOG.md): finalize the 1.0.0 entry — call out the breaking import change (`import pulseqzero as pp`, no more `pp_impl` / `mr0_mode`) and the new `.seq` export path. Note that there is no deprecation shim: scripts migrating from 0.3 need the two mechanical edits.

## Post-unification follow-ups (not part of this PR)

- Pulse-shape autograd. Re-enable per-factory via an opt-in `differentiable=True` flag if anyone needs to optimize TBW / apodization / shape duration.
- Arbitrary-gradient waveform autograd. Only the amplitude scale is differentiable today; the waveform samples are not.
- PyPulseq-1.5 coverage audit. New kwargs (`freq_ppm`, `phase_ppm`, `center`, `phase_modulation`, `oversampling`), new functions (`make_soft_delay`, `calc_adc_segments`, `enable_trace`/`disable_trace`), new `Sequence` methods (`find_block_by_time`, `apply_soft_delay`). `Opts.B0` / `adc_samples_limit` / `adc_samples_divisor` come along for free once §2 lands.
- Translation caching. If the `to_pypulseq()` warning starts firing inside hot loops, cache the translated PyPulseq sequence on the adapter `Sequence` with invalidation on `add_block` / `set_definition`.
