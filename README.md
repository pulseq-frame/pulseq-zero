# Pulseq-zero

Pulseq-zero allows to define MRI sequences with the Python[^1] port of Pulseq[^2]: PyPulseq[^3], and use them within MR-zero[^4].
This way they are deeply integrated in a differentiable digital twin, enabling not only the simulation of the defined sequence but also the efficient optimization of any sequence parameter and any loss function, using the power of PyTorch[^5] and gradient-descent with backpropagation.

Pulseq-zero uses PDG[^6], a fast, analytical and physically exact simulation model that calculates signals that are comparable to in-vivo measurements within seconds.
At the same time, the required changes to the sequence code are minimal; Pulseq-zero exports the optimized sequence works by simply using the installed PyPulseq without any interference.


## Table of contents

1. [General Information](#1-general-information)
2. [Usage](#2-usage)
3. [Development](#3-development)
4. [API](#4-api)
5. [References](#5-references)


## 1. General Information

Pulseq-zero can be cloned from this repository but is also hosted on [PyPI](https://pypi.org/project/pulseqzero/), install it locally with:
```bash
pip install pulseqzero
```

> [!NOTE]
> Pulseq-zero does not declare any runtime dependencies, but it expects `pypulseq`, `torch`, `MRzeroCore`, `numpy`, and `matplotlib` to already be in the environment.
> Starting with 1.0 it targets **PyPulseq 1.5.0.post1**; earlier 1.4.x scripts may need small adjustments where the pypulseq API changed.
>
> **Migration from 0.x:** the mode-switching facade has been removed. Replace `import pulseqzero; pp = pulseqzero.pp_impl` with `import pulseqzero as pp`, and drop any `with pulseqzero.mr0_mode():` wrappers — `seq.to_mr0()` and `seq.write()` both work unconditionally now.

Pulseq-zero was displayed at [ESMRMB 2024](https://www.esmrmb2024.org/)!
You can view the abstract here: [abstract/abstract.md](abstract/abstract.md).
This project is affiliated with MR-zero and PDG but none of the other technologies.
It relies on the following amazing projects:
- [Python](https://www.python.org/) is the programming language used for Pulseq-zero
- [Pulseq](https://pubmed.ncbi.nlm.nih.gov/27271292/) is a vendor-agnostic library and file format for sequence definition and transfer to real systems
- [PyPulseq](https://joss.theoj.org/papers/10.21105/joss.01725) is the port of Pulseq to Python
- [MR-zero](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28727) is a digital twin of the full measurement and reconstruction pipeline for sequence optimization and discovery
- [PyTorch](https://arxiv.org/abs/1912.01703) is an ecosystem of tools for efficient tensor math with GPU accelleration, autograd through backpropagation and a wide variety of optimizers
- [PDG](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30055) (short for Phase Distribution Graphs) is a state-of-the-art Bloch simulation that produces accurate MRI signals for any sequence, orders of magnitude faster than other approaches


## 2. Usage

Pulseq-zero is a drop-in replacement for PyPulseq: any existing PyPulseq script runs under pulseq-zero by swapping the import. The same script can then write `.seq` files *and* be consumed differentiably by MR-zero — no context managers, no mode flags.

```python
import pulseqzero as pp

# Build the sequence exactly like a PyPulseq script.
seq = pp.Sequence()
seq.add_block(pp.make_delay(10e-3))
```

### Define the sequence as a function

Wrap the sequence code in a function so the same definition can drive both `.seq` export and MR-zero simulation / optimization:

```python
def my_gre_seq(TR, TE):
    seq = pp.Sequence()
    # ... create your sequence ...
    seq.add_block(pp.make_delay(TR - 3e-3))
    # ... more sequence creation ...
    return seq
```

### Application

- **Export a `.seq` file and plot** (goes through PyPulseq under the hood; a one-off translation warning is emitted so you notice if it fires inside a hot loop):
  ```python
  seq = my_gre_seq(14e-3, 5e-3)
  seq.plot()
  seq.write("tse.seq")
  ```
- **Simulate with MR-zero**:
  ```python
  import MRzeroCore as mr0

  seq = my_gre_seq(14e-3, 5e-3).to_mr0()
  graph = mr0.compute_graph(seq, sim_data)
  signal = mr0.execute_graph(graph, seq, sim_data)
  reco = mr0.reco_adjoint(signal, seq.get_kspace())
  ```
- **Optimize sequence parameters with PyTorch**:
  ```python
  TR = torch.tensor(14e-3, requires_grad=True)
  TE = torch.tensor(5e-3, requires_grad=True)
  optimizer = torch.optim.Adam([TR, TE], lr=0.001)

  for _ in range(100):
      optimizer.zero_grad()
      seq = my_gre_seq(TR, TE).to_mr0()
      loss = my_loss(seq)
      loss.backward()
      optimizer.step()

  # After optimization: export using the same script.
  my_gre_seq(TR, TE).write("tse_optim.seq")
  ```


## 3. Development

If you want to contribute to Pulseq-zero or make local changes to it, the easiest way is to install it locally in editable mode:

1. Create a virtual environment that can use globally installed packages
  ```bash
  python -m venv --system-site-packages .venv
  ```
2. Activate this environment
  ```bash
  # Windows CMD
  .venv\Scripts\activate
  # Linux bash
  $ source .venv/bin/activate
  ```
3. Install pulseq-zero in the virtual enviornment in editable mode
  ```bash
  pip install --editable .
  ```


## 4. API

### Differentiable rounding

PyPulseq aligns many events to the block / gradient / ADC raster, which requires rounding — and rounding kills gradients. Pulseq-zero ships `pp.round` / `pp.ceil` / `pp.floor` that match PyTorch semantics but act like the identity function on the backward pass:

```python
my_param = torch.tensor(1.5, requires_grad=True)
some_calc = pp.round(torch.sin(my_param))
some_calc.backward()

assert some_calc == 1
assert my_param.grad == torch.cos(my_param)
```

Use these whenever you round a timing (or any sequence quantity) that flows from an optimization parameter. For plain numeric rounding outside optimization, `np.round` / `torch.round` are fine.

### `seq.to_mr0()` and `seq.write()`

Every `pulseqzero.Sequence` supports both paths unconditionally:

- `mr0_seq = seq.to_mr0()` — build an `MRzeroCore.Sequence` for PDG simulation / optimization.
- `seq.write("out.seq")` — translate the internal event graph through PyPulseq and emit a `.seq` file. A one-time `warnings.warn` is raised per call so you notice if it fires inside a hot loop (move it out of the optimizer).

If you need a native PyPulseq `Sequence` for a one-off exotic call, `seq.to_pypulseq()` is the explicit escape hatch.

### PyPulseq coverage (1.5.0.post1)

Pulseq-zero covers what's needed to run differentiable simulation / optimization and to emit `.seq` files. The table below tracks per-function status.

Legend:
- ✅ differentiable native implementation in the adapter.
- ➡️ forwarded to PyPulseq at export / call time (values stay numeric).
- 🚫 raises `NotImplementedError` with a named workaround.


| PyPulseq entry point                   | status | notes |
| -------------------------------------- | ------ | ----- |
| `Sequence.__init__`, `add_block`, `set_definition`, `get_definition`, `duration`, `__str__`, `remove_duplicates` | ✅ | adapter-native |
| `Sequence.to_mr0`                      | ✅ | adapter-native, only on pulseq-zero |
| `Sequence.write`, `to_pypulseq`        | ➡️ | lazy translation, one warning per call |
| `Sequence.check_timing`                | ⚠️ | stub returns `(True, [])` — real validation happens on `write()` |
| `Sequence.plot`, `test_report`, `calculate_pns`, `paper_plot` | ➡️ | forwarded via `to_pypulseq()` |
| `calc_SAR`, `make_label`               | stub | no-op |
| `calc_rf_bandwidth`, `calc_rf_center`  | stub | numeric approximations; pulse shape detail not tracked |
| `calc_duration`                        | ✅ | differentiable, torch.maximum-based |
| `Opts`                                 | ➡️ | direct re-export of `pypulseq.Opts` (all fields) |
| `make_trapezoid`, `make_extended_trapezoid`, `make_arbitrary_grad`, `add_gradients`, `scale_grad`, `split_gradient`, `split_gradient_at` | ✅ | adapter-native, differentiable |
| `make_extended_trapezoid_area`         | ⚠️ | copied from PyPulseq; **not yet differentiable** |
| `make_sinc_pulse`, `make_gauss_pulse`, `make_block_pulse`, `make_arbitrary_rf` | ✅ | delegate shape generation to PyPulseq, keep differentiable `flip_angle` / `phase_offset` / `freq_offset` / `delay` |
| `make_adc`, `make_delay`, `make_trigger`, `make_digital_output_pulse` | ✅ | adapter-native |
| `get_supported_labels`                 | ✅ | static list |
| `make_adiabatic_pulse`, `sigpy_n_seq`, `make_slr`, `make_sms`, `SigpyPulseOpts` | 🚫 | use PyPulseq directly, wrap the signal via `make_arbitrary_rf` |
| `align`, `calc_ramp`, `rotate`, `points_to_waveform`, `traj_to_grad` | 🚫 | not wired yet; call via `seq.to_pypulseq()` if needed |

### Differentiability

Gradients flow through the following quantities end-to-end (set `requires_grad=True` and they thread through to `seq.to_mr0()`):

- RF `flip_angle`, `phase_offset`, `freq_offset`, `delay`
- ADC `phase_offset`, `freq_offset`, `delay`, `dwell`
- Gradient `amplitude` (trapezoidal and arbitrary), `rise_time`, `flat_time`, `fall_time`, `delay`
- Block / repetition / TR / TE durations

The following are **not** differentiable today (they affect pulse *shape*, which is materialized eagerly via PyPulseq):

- Pulse shape parameters: `duration` when used to shape the envelope, `time_bw_product`, `apodization`, `center_pos`, `slice_thickness` (as it feeds shape generation), `dwell` for pulses
- Gradient *waveform samples* for arbitrary gradients (the scale is differentiable, the samples aren't)
- `Opts` fields (max_grad, rasters, dead times) — intentionally numeric

Pulse-shape autograd can be added back per-factory via an opt-in flag if it ever becomes load-bearing.


## 5. References

[^1]: python programming language: https://www.python.org/
[^2]: Layton K et al: Pulseq: A rapid and hardware-independent pulse sequence prototyping framework. MRM 2017, [doi: 10.1002/mrm.26235](https://pubmed.ncbi.nlm.nih.gov/27271292/)
[^3]: Keerthi SR et al: PyPulseq: A Python Package for MRI Pulse Sequence Design. JOSS 2019, [doi: 10.21105/joss.01725](https://joss.theoj.org/papers/10.21105/joss.01725)
[^4]: Loktyushin A et al: MRzero - Automated discovery of MRI sequences using supervised learning. MRM 2021, [doi: 10.1002/mrm.28727](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28727)
[^5]: Paszke A et al: PyTorch: An Imperative Style, High-Performance Deep Learning Library. arxiv 2019, [doi: 10.48550/arXiv.1912.01703](https://arxiv.org/abs/1912.01703)
[^6]: Endres J et al: Phase distribution graphs for fast, differentiable, and spatially encoded Bloch simulations of arbitrary MRI sequences. MRM 2024, [doi: 10.1002/mrm.30055](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30055)
