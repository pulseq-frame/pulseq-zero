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
> Pulseq-zero does not define or require any dependencies.
> That being said, it models the API of PyPulseq 1.4.2 and is only tested with that version of PyPulseq;
> Using it for scripts that are written for other versions of PyPulseq might lead to unexpected results or errors if function names, arguments or behaviour changed.

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

Example scripts are provided in the [tests](tests) folder.
They are modified versions of the PyPulseq 1.4.2 examples found [here](https://github.com/imr-framework/pypulseq/tree/v1.4.2/pypulseq/seq_examples/scripts).
The changes that are typically necessary to convert from a pypulseq sequence script to Pulseq-zero are as follows:

### Import pulseq

Change the python imports to access all functions via the Pulseq-zero facade:
A wrapper that can switch between the pulseq - MR-zero interface and the real pypulseq.

Before:
```python
import pypulseq as pp

# Build a sequence...
seq = pp.Sequence()
seq.add_block(pp.make_delay(10e-3))
```

After:
```python
import pulseqzero
pp = pulseqzero.pp_impl
  
# Use exactly as before...
seq = pp.Sequence()
seq.add_block(pp.make_delay(10e-3))
```

### Define the sequence

Wrap the sequence code in a function.
This is not a necessity but a best practice for better code organization and done in newer pypulseq examples as well.
Namely, it allows to:

 - switch executing the sequence script with pypulseq and write a .seq file and simulation with MR-zero
 - define sequence parameter as function arguments for re-creating the sequence with different settings
 - easily use the sequence definition in an optimization loop.

The result is something like the following example:

```python
def my_gre_seq(TR, TE):
  seq = pp.Sequence()

  # ... create your sequence ...
  seq.add_block(pp.make_delay(TR - 3e-3)) # use the parameters in any way
  # ... more sequence creation ...

  return seq
```

### Application

The sequence definition can now be used in many ways!

- Using with pulseq for plotting and exporting:
  ```python
  seq = my_gre_seq(14e-3, 5e-3)
  seq.plot()
  seq.write("tse.seq")
  ```
- Using with MR-zero for simulation:
  ```python
  import MRzeroCore as mr0
  # Data loading and other imports
  
  with pulseqzero.mr0_mode():
    seq = my_gre_seq(14e-3, 5e-3).to_mr0()

  graph = mr0.compute_graph(seq, sim_data)
  signal = mr0.execute_graph(graph, seq, sim_data)
  reco = mr0.reco_adjoint(signal, seq.get_kspace())
  ```
- Using pulseq-zero helpers to simplify common tasks even more!
  ```python
  # Define some target_image which we try to achieve
  
  TR = torch.tensor(14e-3)
  TE = torch.tensor(5e-3)
  for iter in range(100):
    pulseqzero.optimize(my_gre_seq, target_image, TR, TE)

  # Back to using plain old pypulseq for export again!
  # The pulseq-zero magic is disabled outside of all special calls but theparameters remain optimized
  seq = my_gre_seq(TR, TE)
  seq.write("tse_optim.seq")
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

### Additional API

Pulseq requires many events to align with the block-, gradient- or ADC raster.
For this, rounding is necessary, which is not differentiable.
As a workaround, Pulseq-zero includes differentiable rounding functions, which are wrappers around the corresponding PyTorch functions, but which behave like the identity function for backpropagation.
This means they are invisible to the calculation of gradients, which is okay as long as the rounding is small compared to the changes to the optimized parameters, which is typically the case with small raster times.

```python
my_param = torch.tensor(1.5, requires_grad=True)
some_calc = pp.round(torch.sin(my_param))
some_calc.backward()

assert some_calc == 1
assert my_param.grad == torch.cos(my_param)
```
Use pulseq-zeros rounding functions instead of e.g. those of numpy or torch if it is applied to timings (or other sequence properties) that are part of an optimization.

In the `mr0_mode` context, all PyPulseq functions are swapped for Pulseq-zero implementations that track all calls so that the sequence can be converted to MR-zero later:
```python
with pulseqzero.mr0_mode():
  seq = build_my_seq()
```

If you use some functions that are only available in PyPulseq but not Pulseq-zero (PNS computations or similar), you can check if `mr0_mode` is activated:
```python
if pulseqzero.is_mr0_mode():
  pass
else:
  seq.calculate_pns()
```

Finally, the sequence object returned in `mr0_mode` has one method that is not available otherwise and forms the central point of Pulseq-zero:
```python
with pulseqzero.mr0_mode():
  seq = build_my_seq()
  # This function here
  mr0_seq = seq.to_mr0()
  # Now do some simulations with mr0_seq!
```


### PyPulseq API

The following is a list of all functions and methods exposed in PyPulseq 1.4.2 when importing it directly.
Pulseq-zero tries to provide all of those - it is currently not planned to cover other functions that are available internally in PyPulseq but not exposed this way.
If you need one of them, file an issue or submit a pull request.

Not all functions on this list will be supported: Pulseq-zero does not aim to translate functions that are not differentiable or would require to emulate large portions of inner workings of PyPulseq if they are rarely used, other functions will not do any actual work (like test reports) as they are not useful when simulating / optimizing sequences.
The list tracks the progress on deciding which function to support and implementing them if desired.
Functions that behave differently to PyPulseq are listed in the following sections.


- [x] `calc_SAR`
- [ ] `Sequence`
  - [x] `__init__`
  - [x] `__str__`
  - [ ] `adc_times`
  - [x] `add_block`
  - [ ] `calculate_gradient_spectrum`
  - [ ] `calculate_kspace`
  - [ ] `calculate_kspacePP`
  - [x] `calculate_pns`
  - [x] `check_timing`
  - [x] `duration`
  - [x] `evaluate_labels`
  - [ ] `flip_grad_axis`
  - [ ] `get_block`
  - [x] `get_definition`
  - [ ] `get_extension_type_ID`
  - [ ] `get_extension_type_string`
  - [ ] `get_gradients`
  - [ ] `mod_grad_axis`
  - [x] `plot`
  - [x] `read`
  - [x] `register_adc_event`
  - [x] `register_grad_event`
  - [x] `register_label_event`
  - [x] `register_rf_event`
  - [x] `remove_duplicates`
  - [ ] `rf_from_lib_data`
  - [ ] `rf_times`
  - [ ] `set_block`
  - [x] `set_definition`
  - [ ] `set_extension_string_ID`
  - [x] `test_report`
  - [ ] `waveforms`
  - [ ] `waveforms_and_times`
  - [ ] `waveforms_export`
  - [x] `write`
- [x] `add_gradients`
- [ ] `align`
- [x] `calc_duration`
- [ ] `calc_ramp`
- [x] `calc_rf_bandwidth`
- [x] `calc_rf_center`
- [x] `make_adc`
- [x] `make_adiabatic_pulse`
- [x] `make_arbitrary_grad`
- [x] `make_arbitrary_rf`
- [x] `make_block_pulse`
- [x] `make_delay`
- [x] `make_digital_output_pulse`
- [x] `make_extended_trapezoid`
- [x] `make_extended_trapezoid_area`
- [x] `make_gauss_pulse`
- [x] `make_label`
- [x] `make_sinc_pulse`
- [x] `make_trapezoid`
- [x] sigpy
  - [x] `SigpyPulseOpts`
  - [x] `sigpy_n_seq`
  - [x] `make_slr`
  - [x] `make_sms`
- [x] `make_trigger`
- [x] `Opts`
  - [x] `__init__`
  - [x] `set_as_default`
  - [x] `reset_default`
  - [x] `__str__`
- [x] `points_to_waveform`
- [ ] `rotate`
- [x] `scale_grad`
- [x] `split_gradient`
- [ ] `split_gradient_at`
- [x] `get_supported_labels`
- [ ] `traj_to_grad`

### Disabled in mr0 mode

These will in mr0 mode either return the specified value or don't exist so using them raises an error.
The reason for disabling is either that a differentiable re-implementation is out of scope of pulseq-zero (e.g.: pulse optimization) or not sensible (like sequence loding).

| function | return value |
| -------- | ------------ |
| `calc_SAR` | `(True, [])` |
| `Sequence.calculate_pns` | **error** |
| `Sequence.check_timing` | `None` |
| `Sequence.evaluate_labels` | **error** |
| `Sequence.plot` | `None` |
| `Sequence.read` | **error** |
| `Sequence.write` | `None` or `""` depending on `create_signature` |
| `Sequence.register_*_event` | **error** - is only used internally |
| sigpy | **error** |
| `make_adiabatic_pulse` | **error** |
| `make_label` | `None` |
| `make_extended_trapezoid_area` | **error** |
| `calc_rf_bandwidth` | returns zeroes as mr0 mode doesn't store pulse shapes |
| `calc_rf_center` | returns (shape_dur / 2, 0) - mr0 mode ignores (assymetric) pulse shapes |
| `points_to_waveform` | **error** |

### Altered behaviour

Some functions are only partially supported in mr0 mode and / or some aspects are missing in simulation.
In general, pulseq-zero tries not to include every single attribute that exists in pypulseq to reduce bloat; if scripts rely on them existing (even if they are ignored even in pypulseq) they can be added to the objects created in the `make_` functions even if they don't affect anything.
Some pypulseq functions round to raster times, which is not done in mr0 mode for differentiability.
Pypulseq does many more checks on timing or other parameters that are not checked by pulseq-zero.
These are not listed here, but note that some scripts that don't run otherwise might run in mr0 mode.
Pulses have no shape, `center_pos` will influence the returned `gzr` but nothing else - *TODO* should be considered when converting timing to mr0

| function | remarks |
| -------- | ------- |
| `make_trigger`, `make_digital_output_pulse` | returns `Delay`, ignores rest |
| `make_adc` | has no `dead_time` property |
| `make_arbitrary_rf`, `make_block_pulse`, `make_gauss_pulse`, `make_sinc_pulse` | returned object has no `signal` or `t` attribute (waveform is not computed) but has an added `flip_angle` |
| `make_trapezoid` | `area`, `flat_area` are calculated properties, no attributes `first`, `last` or `use`, `signal` or `t` |
| `make_sinc_pulse` | has no `dead_time`, `ringdown_time` or `use` properties |
| `make_gauss_pulse` | has no `dead_time`, `ringdown_time`, `use`, `signal` or `t` |
| `make_arbitrary_rf` | has no `dead_time`, `ringdown_time`, `use`, `signal` or `t` |
| `calc_duration` | adc doesn't include dead_time (pypulseq bug), only includes trigger delay (see `make_trigger`) |
| `make_arbitrary_grad` | no `first` or `last` |
| `make_extended_trapezoid` | skipping some checks and ignoring `convert_to_arbitrary`; `area` is a computed property, no `first` or `last` |
| `make_extended_trapezoid_area` | Implementation copied from pypulseq, not yet differentiable (can't optimize parameters of gradients created with this function) |
| `add_gradients` | Implementation copied from pypulseq, not yet differentiable (can't optimize parameters of gradients created with this function) |
| `Sequence` | way less internal bookkeeping, most variables are missing, reports etc. are not calculated |


## 5. References

[^1]: python programming language: https://www.python.org/
[^2]: Layton K et al: Pulseq: A rapid and hardware-independent pulse sequence prototyping framework. MRM 2017, [doi: 10.1002/mrm.26235](https://pubmed.ncbi.nlm.nih.gov/27271292/)
[^3]: Keerthi SR et al: PyPulseq: A Python Package for MRI Pulse Sequence Design. JOSS 2019, [doi: 10.21105/joss.01725](https://joss.theoj.org/papers/10.21105/joss.01725)
[^4]: Loktyushin A et al: MRzero - Automated discovery of MRI sequences using supervised learning. MRM 2021, [doi: 10.1002/mrm.28727](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28727)
[^5]: Paszke A et al: PyTorch: An Imperative Style, High-Performance Deep Learning Library. arxiv 2019, [doi: 10.48550/arXiv.1912.01703](https://arxiv.org/abs/1912.01703)
[^6]: Endres J et al: Phase distribution graphs for fast, differentiable, and spatially encoded Bloch simulations of arbitrary MRI sequences. MRM 2024, [doi: 10.1002/mrm.30055](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30055)
