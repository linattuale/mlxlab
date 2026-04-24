# Changelog

All notable changes to `mlxlab` will be documented in this file.

The project is currently pre-1.0, so APIs may still change as the library
hardens.

## [Unreleased]

### Added

- Add an opt-in mixed-precision integration path with `solve(..., rhs_dtype=...)`
  and `integrate.mixed_matmul(...)` for matmul-heavy RHS functions

### Changed

- Require `mlx>=0.31.1` so supported installs include the current M5-tuned MLX runtime line
- Chunk fixed-step ODE and SDE integration into compiled MLX blocks to reduce per-step dispatch
- Compile adaptive RK trial steps and step-size decisions so Python only observes accepted time
  advances
- Vectorize `welch`, `spectrogram`, and `saveat` interpolation to avoid serial Python window/index
  loops
- `_solve_fixed` now reconstructs output `t` as `t0 + i * dt` rather than accumulating via
  repeated `t += dt`, eliminating floating-point drift in reported times
- `psd`, `welch`, and `spectrogram` now reject non-1-D input explicitly instead of producing
  silently malformed results
- Document MLX ecosystem assumptions as of Apr 24, 2026, including M5, Metal 4,
  Neural Accelerator, and CUDA-backend caveats

### Notes

- SDE trajectories under `method="euler_maruyama"` will not be bit-identical to 0.1.1 for the
  same seed once `n_steps >= 16`: chunked integration batches the noise draws as a single
  `mx.random.normal((16,) + y.shape)` per chunk instead of one draw per step, which reorders
  the PRNG stream. Distributional properties are unchanged.

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
