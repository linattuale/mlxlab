# Roadmap

This roadmap is intentionally opinionated: it reflects what would most improve
`mlxlab` for real users, not just what would add the most surface area.

## External Context (as of Apr 24, 2026)

MLX does not appear to have a single formal public roadmap. For planning,
`mlxlab` tracks MLX release notes, issues, pull requests, and discussions. The
important current signals are:

- M5 Neural Accelerator support exists in the MLX v0.30+ runtime line.
- MLX custom Metal kernels are available through `mx.fast.metal_kernel`, but
  direct Neural Accelerator programming is not exposed as a normal custom-kernel
  target.
- Metal 4 machine-learning passes are model/package oriented, not a direct
  replacement for MLX array kernels in scientific solvers.
- MLX has Linux/NVIDIA CUDA packages (`mlx[cuda12]`, `mlx[cuda13]`) and custom
  CUDA kernels, but `mlxlab` should treat CUDA as unvalidated until its
  integrate, signal, random, and linalg modules have backend-specific tests and
  benchmarks.

## Highest Priority

### 1. Compiled integration loop

The biggest remaining performance limitation is the Python-level accept/reject
loop in adaptive `mlxlab.integrate` solvers. Fixed-step solvers now run in
compiled chunks, and adaptive trial steps compile the RHS, error estimate, and
step-size decision. The next step is a fully compiled adaptive loop that can
record accepted states without returning to Python on every trial.

### 2. More examples and user-facing docs

The library now spans `integrate`, `linalg`, `signal`, and `random`, but the
strongest public story is still `integrate`. More worked examples would help
users understand where `mlxlab` is already useful today.

### 3. Hardening and comparison tests

The next credibility gains are likely to come from:

- more SciPy/NumPy parity tests for `signal` and `linalg`
- more statistical and keyed-RNG tests for `random`
- cleaner CI and environment reproducibility for contributors

### 4. Mixed-precision integration (FP16/BF16 matmuls)

M5 GPU has dedicated matrix multiply hardware that runs FP16 at ~70 TFLOPS —
much faster than FP32 on general ALUs. The integrate solvers currently run
state, error estimates, and step sizes in FP32 by default, and now expose an
opt-in `rhs_dtype` path plus `mixed_matmul` helper so matmul-heavy RHS functions
can run their `W @ r` work through FP16 or BF16 inputs. Remaining work is
turning the quick N=2000-8000 spot checks into a reproducible benchmark sweep,
measuring accuracy/performance tradeoffs across large chaotic systems, and
deciding whether any automatic problem-size heuristics are worth the added API
complexity.

### 5. Direct Metal 4 / Neural Accelerator path

mlxlab currently relies on MLX to select the best Metal kernels. It does not
directly program M5 GPU Neural Accelerators or Metal 4 Tensor APIs. That keeps
the library simple and Pythonic, but the most hardware-specific M5 gains would
require either upstream MLX coverage or a lower-level kernel layer.

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
