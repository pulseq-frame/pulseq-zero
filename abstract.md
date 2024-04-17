# Pulseq-zero: Automatic optimization of pulseq MRI sequences


## Summary (290 / 300 characters)

We combine our efficient and fully differentiable MR simulation with a modified version of pypulseq to allow optimization of any pulseq sequence script. Requiring nothing more than the difinition of optimization parameters and a loss function, any imaging properties can be achieved easily.

Moritz:
"MRzero turns seq files into an complete and optimizable MRI sequence and reconstruction operator"


## Abstract (3197 / 5000 characters)

### Introduction (644 characters)
MRI sequences are often defined in terms of many parameters, whose values are often highly customizable without a clear optimal choice. MR-zero provides a solution to this with its fast and fully differentiable simulation, by allowing fast optimization of flip angles, inversion times, timings, and more. Until now, it used its own internal sequence definition. With pulseq-zero we expand MR-zero to deeply integrate with the widely adopted pulseq standard. This enables optimization of any pulseq script, only requiring minimal modifications and not manipulating the export and measurement of the programmed and optimized sequences in any way.

### Methods (1705 characters)
Pulseq-zero provides a swappable interface that is fully compatible with pypulseq. Existing pypulseq scripts only need to swap imports and then be modified to be built based on the optimized parameters, as can be seen in [Figure 1](asdf.png). When executing the script normally, the functions are still plain pypulseq and the exported .seq file is identical to before. When using pulseq-zeros optimization methods, the implementation is swapped and pypulseq is replaced with pulseq-zeros implementation that tracks all parameters. The resulting sequence is then converted internally to the format required by MR-zero while tracking all operations throughout the process to remain fully differentiable. As a result, only a loss function needs to be added to the pypulseq script in order to optimize it with respect to any target.

To demonstrate the optimization, two examples are tested: The adjustment of the inversion time for optimal fluid supression of an EPI sequence, as well as the optimization of refocusing flip angles with the goal of achieving a synthetic target contrast without blurring. Figure 1 shows the code changes to the official pypulseq EPI sequence; besides changing imports and adding an inversion rf pulse in the beginning, no changes were made to the script. The subsequent optimization only defines the loss function, which maximises the difference of the average signal to a selected CSF voxel. Figure 2 shows the code changes to the official pypulseq TSE sequence; It was written to correctly utilize the refocusing pulse flip angle array. Afterwards, a synthetic contrast with hardcoded T1 and T2 weighting was used and the flip angles optimized, to achieve a similar image.

### Results (443 characters)
As can be seen in Figures 3 and 4, both optimizations succeeded. As the sequences itself were never modified, only their parameters were altered, they still function on all systems as before, only with optimized contrasts. Optimization times were reasonable at ? s / iteration for EPI and ? s / iteration for TSE, without any adjustments to maximize speed and calculated on a mid-range CPU. Simulation results are proven to be accurate [refs].

### Discussion & Conclusions (405 characters)

The advantage of this approach is that as the original pypulseq script is intact, no roundtrip between real scanner definition and simulation definition is necessary, and all the scanner specific limitations etc. are kept intact. In addition, pypulseq is a widely used standard, nearly no training is necessary in order to be able to optimize pypulseq scripts and use them in real measurements afterwards.


## References

