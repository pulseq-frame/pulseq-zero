# Pulseq-zero

This project aims to join pulseq with MR-zero:
Use pypulseq for sequence programming and insert the sequence definition directly in MR-zero for simulations, optimization and more!


Built for [pypulseq 1.4.2](https://github.com/imr-framework/pypulseq/tree/v1.4.2)


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
    # The pulseq-zero magic is disabled outside of all special calls but the parameters remain optimized
    seq = my_gre_seq(TR, TE)
    seq.write("tse_optim.seq")
  ```

## API

The following pypulseq methods and classes currently exist in Pulseq-zero.
If your sequence scripts rely methods that are not listed here, Pulseq-zero might not yet be usable.
Please create an issue with the request for the missing functionality.
