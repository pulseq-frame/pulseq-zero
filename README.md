# Pulseq-zero

This project aims to join pulseq with MR-zero:
Use pypulseq for sequence programming and insert the sequence definition directly in MR-zero for simulations, optimization and more!


Built for [pypulseq 1.4.2](https://github.com/imr-framework/pypulseq/tree/v1.4.2)


## Development

1. Create a virtual environment that can use globally installed packages
  ```bash
  python -m venv --system-site-packages .venv
  ```
2. Activate this environment
  ```bash
  .venv\Scripts\activate
  ```
3. Install pulseq-zero in the virtual enviornment in editable mode
  ```bash
  pip install --editable .
  ```


## Usage

Example scripts are provided in 'pulseqzero/seq_examples'.
They are modified versions of the pypulseq 1.4.2 examples.
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
pp = pulseqzero.pp_facade
  
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
    seq = my_gre_seq()

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

## API

The following pypulseq methods and classes currently exist in Pulseq-zero.
If your sequence scripts rely methods that are not listed here, Pulseq-zero might not yet be usable.
Please create an issue with the request for the missing functionality.

Pypulseq 1.4.2 has the following re-exports that are expected to be used frequently.
They fall into different categories as specified below, which also are the roadmap for pulseq-zero development.

```python
from pypulseq.SAR.SAR_calc import calc_SAR
from pypulseq.Sequence.sequence import Sequence
from pypulseq.add_gradients import add_gradients
from pypulseq.align import align
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_ramp import calc_ramp
from pypulseq.calc_rf_bandwidth import calc_rf_bandwidth
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_adc import make_adc
from pypulseq.make_adiabatic_pulse import make_adiabatic_pulse
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_sigpy_pulse import *
from pypulseq.make_delay import make_delay
from pypulseq.make_digital_output_pulse import make_digital_output_pulse
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_label import make_label
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts
from pypulseq.make_trigger import make_trigger
from pypulseq.opts import Opts
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.rotate import rotate
from pypulseq.scale_grad import scale_grad
from pypulseq.split_gradient import split_gradient
from pypulseq.split_gradient_at import split_gradient_at
from pypulseq.supported_labels_rf_use import get_supported_labels
from pypulseq.traj_to_grad import traj_to_grad
```

### TODO

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
- [ ] `add_gradients`
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
The reason for disabling is either that a differentiable re-implementation if out of scope of pulseq-zero (e.g.: pulse optimization) or not sensible (like sequence loding).

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
| `calc_rf_center` | returns zeroes as mr0 mode doesn't store pulse shapes |
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
| `Sequence` | way less internal bookkeeping, most variables are missing, reports etc. are not calculated |

### Additional API

This API does not exist in pulseq and is provided for simulation, optimization and other tasks.

