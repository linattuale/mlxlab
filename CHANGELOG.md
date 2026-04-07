# Changelog

All notable changes to `mlxlab` will be documented in this file.

The project is currently pre-1.0, so APIs may still change as the library
hardens.

## [Unreleased]

- No unreleased changes yet.

## [0.1.1] - 2026-04-07

### Added

- `mlxlab.linalg` with `det`, `slogdet`, `lstsq`, `matrix_rank`, and `cond`
- `mlxlab.signal` with `fftfreq`, `rfftfreq`, `psd`, `welch`, and `spectrogram`
- `mlxlab.random` with `exponential`, `gamma`, `beta`, `poisson`, and `binomial`

### Changed

- Hardened `mlxlab.integrate` input validation and `saveat` behavior
- Clarified benchmark methodology and README positioning for public release
- `mlxlab.linalg.lstsq` now returns the minimum-norm least-squares solution
- `mlxlab.random.poisson` and `mlxlab.random.binomial` now return `int32`

### Fixed

- Key reuse bugs in `mlxlab.random` when explicit PRNG keys are provided
- Non-square matrix handling in `mlxlab.linalg.det` and `mlxlab.linalg.slogdet`
- Singular-matrix handling in `mlxlab.linalg.cond`
- Odd-length one-sided PSD scaling in `mlxlab.signal.psd`
- Validation for `welch` and `spectrogram` segment parameters

## [0.1.0] - 2026-04-07

### Added

- Initial public package structure
- `mlxlab.integrate` with fixed-step ODE solvers, adaptive RK solvers, and Euler-Maruyama
- Benchmarks for Dormand-Prince 5(4) across multiple frameworks on Apple Silicon
