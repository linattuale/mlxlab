# Roadmap

This roadmap is intentionally opinionated: it reflects what would most improve
`mlxlab` for real users, not just what would add the most surface area.

## Highest Priority

### 1. Compiled integration loop

The biggest current performance limitation is the Python-level stepping loop in
`mlxlab.integrate`. Moving more of the adaptive stepping logic into compiled MLX
would improve small and medium problem sizes significantly.

### 2. More examples and user-facing docs

The library now spans `integrate`, `linalg`, `signal`, and `random`, but the
strongest public story is still `integrate`. More worked examples would help
users understand where `mlxlab` is already useful today.

### 3. Hardening and comparison tests

The next credibility gains are likely to come from:

- more SciPy/NumPy parity tests for `signal` and `linalg`
- more statistical and keyed-RNG tests for `random`
- cleaner CI and environment reproducibility for contributors

### 4. Mixed-precision integration (FP16 matmuls)

M5 GPU has dedicated matrix multiply hardware that runs FP16 at ~70 TFLOPS —
much faster than FP32 on general ALUs. The integrate solvers currently run
everything in FP32. A mixed-precision mode could cast large matmuls (e.g.,
`W @ r`) to FP16 with FP32 accumulation while keeping solver state (y, error
estimates, step sizes) in FP32 for numerical stability. This could roughly
double performance at large N where the matmul dominates, without sacrificing
adaptive stepping accuracy. Could be exposed as a `dtype` parameter or handled
automatically based on problem size.

## Medium Priority

### Implicit and stiff solvers

The biggest feature gap in `integrate` is still stiff ODE support. BDF or SDIRK
methods would broaden the class of problems that `mlxlab` can solve credibly.

### Dense output and interpolation

Dense output for adaptive solvers and interpolation utilities would make the
integration API more practical for analysis and downstream pipelines.

### Benchmarks beyond integrate

`integrate` is the clearest benchmark story today. Signal and random benchmarks
would become more useful if they demonstrate a real MLX-at-scale advantage,
rather than just checking a box.

## Longer-Term

### Sparse matrices

Sparse support would unlock a lot, but this depends heavily on MLX foundations.
This is important, but not a near-term “ship quickly” target.

### GPU linalg

Once MLX ships GPU decompositions, `mlxlab.linalg` can become much more than a
gap-filling wrapper layer.

### Special functions

Functions like Bessel, `gammaln`, and related numerics would make `mlxlab` more
useful across statistics and physics workloads, but they should come after core
correctness and performance work.
