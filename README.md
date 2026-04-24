# mlxlab

Scientific computing on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

## Install

```bash
uv add mlxlab
```

## Status

`mlxlab` is alpha software.

- Best-supported module today: `mlxlab.integrate`
- Target platform: Apple Silicon + MLX
- Requires `mlx>=0.31.1` for the current M5-tuned MLX runtime line
- `mlxlab.linalg` currently relies on MLX 0.31 CPU-only decompositions
- `mlxlab.signal` and `mlxlab.random` are early convenience modules, not full SciPy/NumPy replacements

See [CHANGELOG.md](CHANGELOG.md), [ROADMAP.md](ROADMAP.md), and
[CONTRIBUTING.md](CONTRIBUTING.md) for project status and contribution guidance.

## MLX Ecosystem Notes (as of Apr 24, 2026)

- MLX does not appear to publish a single formal roadmap; the best public signals
  are release notes, issues, pull requests, and discussions.
- MLX v0.30 added M5 Neural Accelerator support, and this package now requires
  `mlx>=0.31.1` to stay on the current M5-tuned runtime line.
- MLX supports custom Metal kernels via `mx.fast.metal_kernel`, which is the most
  practical route for fused mlxlab-specific GPU kernels. This is not the same as
  directly programming M5 Neural Accelerators.
- Metal 4 machine-learning passes are model/tensor oriented: useful for running
  Core ML/Metal ML packages on the GPU timeline, but not a drop-in custom kernel
  route for Python scientific solvers.
- MLX has a CUDA backend for Linux/NVIDIA (`mlx[cuda12]` or `mlx[cuda13]`) and a
  custom CUDA-kernel API. `mlxlab` does not currently claim CUDA support or ship
  CUDA benchmarks; backend coverage for FFT-heavy signal code and LAPACK-style
  linalg should be validated separately before presenting CUDA results.

## Quick Start

```python
import mlx.core as mx
import mlxlab as ml

# Define your system: dy/dt = -y
def rhs(y, t):
    return -y

# Solve with adaptive stepping (Tsit5, same family as MATLAB's ode45)
sol = ml.integrate.solve(rhs, mx.array([1.0]), t_span=(0, 5), method="tsit5")

print(sol.y[-1])  # ~ exp(-5) = 0.006738
```

## Why mlxlab?

MLX gives you GPU-accelerated array operations on Apple Silicon. mlxlab builds
scientific computing tools on top -- with ODE/SDE solvers as the flagship
module, plus a small set of linalg, signal, and random utilities that MLX
doesn't ship directly.

```python
# 10,000-neuron rate network -- the W @ r hits the GPU automatically
W = mx.random.normal((10000, 10000)) * 0.01
tau = 0.01

def network_rhs(r, t):
    return (-r + mx.tanh(W @ r)) / tau

sol = ml.integrate.solve(network_rhs, mx.zeros(10000), t_span=(0, 1), dt=0.0001, method="rk4")
```

For matmul-heavy systems, keep the solver state and control flow in float32
while running the dominant `W @ r` multiply through half-precision inputs:

```python
W16 = W.astype(mx.float16)

def network_rhs_mixed(r, t):
    Wr = ml.integrate.mixed_matmul(W16, r, dtype=mx.float16, out_dtype=mx.float32)
    return (-r + mx.tanh(Wr)) / tau

sol = ml.integrate.solve(
    network_rhs_mixed,
    mx.zeros(10000),
    t_span=(0, 1),
    dt=0.0001,
    method="rk4",
)
```

For RHS functions where all state-dependent work can tolerate lower precision,
pass `rhs_dtype=mx.float16` or `rhs_dtype=mx.bfloat16` to cast the state view
seen by `f` as well.

## Solvers

| Method | Type | Use case |
|--------|------|----------|
| `euler` | Fixed-step | Teaching, quick tests |
| `rk4` | Fixed-step | When you know your dt |
| `tsit5` | Adaptive (default) | General non-stiff ODEs |
| `dopri5` | Adaptive | MATLAB ode45 equivalent |
| `euler_maruyama` | Fixed-step SDE | Stochastic systems |

**Note:** The RHS convention is `f(y, t)` (state first), matching dynamical systems
convention and MLX's array-first style. This differs from SciPy's `f(t, y)`.

## Benchmarks

Dormand-Prince 5(4) adaptive solver across 6 frameworks, same tolerances
(rtol=1e-4, atol=1e-6). Problem: chaotic rate network
`dy/dt = (-y + tanh(W @ y)) / tau` with gain g=1.5, b=0 (above the Sompolinsky
chaos threshold). All frameworks save the full trajectory. Benchmarked on
M5 Max (40 GPU cores, 128 GB).

![Benchmark: Dormand-Prince 5(4) scaling on M5 Max](benchmarks/bench_dp5.png)

| N | mlxlab GPU | mlxlab CPU | Julia DP5 | MATLAB ode45 | SciPy RK45 | torchdiffeq MPS |
|---|---|---|---|---|---|---|
| 500 | 0.121s | 0.085s | **0.009s** | 0.016s | 0.034s | 1.361s |
| 1000 | 0.125s | 0.088s | **0.097s** | 0.030s | 0.125s | 1.445s |
| 2000 | **0.123s** | 0.153s | 0.148s | 0.136s | 0.549s | 1.408s |
| 4000 | **0.246s** | 0.682s | 0.665s | 0.857s | 2.262s | 1.504s |
| 8000 | **0.736s** | 2.624s | 2.424s | 3.511s | 9.182s | 2.003s |
| 16000 | **2.842s** | 10.251s | 8.682s | 13.929s | 37.245s | 4.038s |
| 32000 | **10.624s** | 41.012s | 34.497s | 54.881s | -- | 17.052s |

**Key findings:**

- **N >= 2000: mlxlab GPU is fastest.** At N=8000, 3.3x faster than Julia, 4.8x
  faster than MATLAB, 12.5x faster than SciPy.
- **N=32000: mlxlab GPU is 3.3x faster than Julia, 1.6x faster than torchdiffeq.**
- **N < 1000: Julia wins** due to compiled language with zero per-step dispatch
  overhead. mlxlab's remaining small-N floor comes from adaptive accept/reject
  bookkeeping that still crosses the Python boundary once per trial.
- **Step counts are consistent** across frameworks: ~140-180 steps for float32
  solvers, ~300-350 for float64 (SciPy/MATLAB), confirming the same algorithm
  is being solved.
- **SciPy and MATLAB use float64 internally** (SciPy upcasts; MATLAB ode45 operates
  in double precision). This is disclosed, not corrected -- it reflects the real-world
  experience of switching frameworks. MATLAB's higher step count (~2x) is a
  consequence of float64 error estimation at the same tolerances.

**Mixed-precision spot check:** The opt-in `mixed_matmul` path is fastest once
the RHS is dominated by large matrix-vector multiplies. On the same M5 GPU,
quick local checks of the chaotic rate-network RHS showed fixed-step RK4 at
1.4x faster for N=2000, 2.3x for N=4000, and 1.9x for N=8000. Adaptive `dopri5`
with the conservative matmul-only pattern was about break-even at N=2000 and
roughly 1.5x faster at N=4000 and N=8000. Small systems can be slower, and full
accuracy/performance sweeps still need to replace these spot checks.

**Current M5 limits:** mlxlab now uses the M5-tuned MLX runtime line and keeps
more solver/signal work on the GPU, but it is not a direct Metal 4 tensor-kernel
library. Adaptive solvers still cross into Python once per trial for accept/reject
bookkeeping, mixed-precision integration is opt-in through `rhs_dtype` and
`mixed_matmul`, there is no direct Neural Accelerator programming, and
`mlxlab.linalg` remains CPU-bound while MLX decompositions are CPU-only.

**Chaos verification:** The system shows chaotic dynamics at all benchmark sizes.
A 1e-6 perturbation to initial conditions amplifies 43x (N=500), 241x (N=1000),
and 306x (N=2000) over T=1s, consistent with positive Lyapunov exponents (though
this is a finite-time sensitivity check, not a rigorous exponent computation).
The finite-size transition is visible at N=500 (weaker chaos) but all sizes show
irregular, non-periodic dynamics.

![Chaos verification: neuron traces and IC sensitivity](benchmarks/chaos_check.png)

**Methodology:** Each framework was benchmarked in a separate process, run
sequentially (never in parallel). System load was verified idle (>70% CPU idle) via
`top` between each run. Median of 5 runs (3 for N >= 16000), 1 warmup run. All
benchmarks use seed 42, same gain/tau/tolerances, and save the full trajectory.
Note: the random matrix W differs across languages (NumPy PCG64 vs Julia
MersenneTwister vs MATLAB default) so the specific chaotic trajectory differs,
but the statistical properties (spectral radius, chaos strength) are equivalent.
Scripts are in `benchmarks/`.

## Additional Modules

### mlxlab.linalg

Functions that MLX doesn't ship, built on its CPU-only decompositions (lu, qr, svd):

```python
import mlxlab as ml

ml.linalg.det(A)            # determinant (via SVD + LU)
ml.linalg.slogdet(A)        # sign and log-abs-determinant
ml.linalg.lstsq(A, b)       # minimum-norm least-squares (via pseudoinverse)
ml.linalg.matrix_rank(A)    # numerical rank (via SVD)
ml.linalg.cond(A)           # 2-norm condition number (returns inf for singular)
```

Note: MLX 0.31's decompositions are CPU-only. Results live in unified memory and
can be used in subsequent GPU operations. For operations MLX already provides
(eig, svd, cholesky, solve, inv, etc.), use `mlx.core.linalg` directly.

### mlxlab.signal

FFT-based spectral analysis built on `mlx.core.fft`:

```python
import mlxlab as ml

freqs = ml.signal.fftfreq(n, d=1/fs)
freqs, power = ml.signal.psd(x, fs=1000)
freqs, power = ml.signal.welch(x, fs=1000, nperseg=256)
times, freqs, Sxx = ml.signal.spectrogram(x, fs=1000, nperseg=256)
```

### mlxlab.random

Distributions that MLX doesn't ship, built on `mlx.core.random`:

```python
import mlxlab as ml

ml.random.exponential(shape=(1000,), scale=2.0)
ml.random.gamma(shape_param=5.0, scale=2.0, shape=(1000,))
ml.random.beta(a=2.0, b=5.0, shape=(1000,))
ml.random.poisson(lam=3.0, shape=(1000,))       # returns int32
ml.random.binomial(n=20, p=0.3, shape=(1000,))  # returns int32
```

Supports explicit PRNG keys via `key=` for reproducibility (keys are split
internally to ensure independence across draws).

## Roadmap (v0.2)

- **Compiled integration loop** -- fixed-step solvers now run in compiled chunks,
  and adaptive trial steps compile the RHS plus step-size decision. The remaining
  bottleneck is the adaptive accept/reject loop crossing into Python once per
  trial; eliminating that would extend mlxlab's advantage to smaller systems.
- **Sparse matrices** -- no MLX foundation exists yet; major undertaking.
- **Implicit/stiff solvers** -- BDF, SDIRK methods for stiff systems.
- **Special functions** -- bessel, gamma function (may need custom Metal kernels).
- **Interpolation** -- interp1d, splines.
- **Dense output** -- continuous interpolation between solver steps.
- **GPU linalg** -- when MLX ships GPU decompositions.

## Acknowledgments

Built by [@linattuale](https://github.com/linattuale) with
[Claude Opus 4.6 (1M context)](https://claude.ai/claude-code) for implementation,
benchmarking, and iteration. Code review, correctness auditing, and repo hygiene by
[OpenAI GPT-5.4 Codex](https://chatgpt.com/codex) — whose two thorough review passes
caught critical bugs (PRNG key reuse, singular matrix handling, PSD normalization)
and significantly improved the library's reliability before public release.

## License

MIT
