# Changelog

All notable changes to pulseq-zero will be recorded in this file as we work
toward the v1.0.0 release.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- [demo/](demo/): a new `uv`-managed workspace member that targets
  PyPulseq 1.5, exercising pulseq-zero against the newer PyPulseq release
  alongside the existing 1.4.2-based examples in [tests/](tests/).
- [demo/write_tse.py](demo/write_tse.py): TSE sequence script ported from the
  PyPulseq 1.5 upstream example to the pulseq-zero facade. The refocusing
  flip angles are now an input (`refoc_flips`, length `n_echo`, radians) and
  the refocusing pulse is rebuilt per echo so per-echo flips take effect.
- [demo/main.py](demo/main.py): end-to-end demo that simulates a target image
  from a fully-180° TSE refocusing train and then optimizes the refocusing
  flip-angle vector with an RMS image-fidelity loss plus a
  `sum(flips**2)` SAR penalty.
