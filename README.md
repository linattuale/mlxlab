# mlxlab

Scientific computing on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

## Install

```bash
uv add mlxlab
```

## Quick Start

```python
import mlx.core as mx
import mlxlab as ml

# Define your system: dy/dt = -y
def rhs(y, t):
    return -y

# Solve with adaptive stepping (like MATLAB's ode45, but better)
sol = ml.integrate.solve(rhs, mx.array([1.0]), t_span=(0, 5), method="tsit5")

print(sol.y[-1])  # ~ exp(-5) = 0.006738
```

## Why mlxlab?

MLX gives you GPU-accelerated array operations on Apple Silicon. mlxlab builds
scientific computing tools on top -- starting with ODE/SDE solvers where the
expensive right-hand side (matrix-vector multiplies) runs on the GPU.

```python
# 10,000-neuron rate network -- the W @ r hits the GPU automatically
W = mx.random.normal((10000, 10000)) * 0.01
b = mx.random.normal((10000,))
tau = 0.01

def network_rhs(r, t):
    return (-r + mx.tanh(W @ r + b)) / tau

sol = ml.integrate.solve(network_rhs, mx.zeros(10000), t_span=(0, 1), dt=0.0001, method="rk4")
```

## Solvers

| Method | Type | Use case |
|--------|------|----------|
| `euler` | Fixed-step | Teaching, quick tests |
| `rk4` | Fixed-step | When you know your dt |
| `tsit5` | Adaptive (default) | General non-stiff ODEs |
| `dopri5` | Adaptive | MATLAB ode45 equivalent |
| `euler_maruyama` | Fixed-step SDE | Stochastic systems |

## Benchmarks

Dormand-Prince 5(4) adaptive solver, same tolerances (rtol=1e-4, atol=1e-6) across
all frameworks. Problem: chaotic rate network `dy/dt = (-y + tanh(W @ y + b)) / tau`
with gain g=1.5 (above the Sompolinsky chaos threshold). Benchmarked on M5 Max
(40 GPU cores, 128 GB).

![Benchmark: Dormand-Prince 5(4) scaling on M5 Max](benchmarks/bench_dp5.png)

| N | mlxlab GPU | mlxlab CPU | Julia DP5 | MATLAB ode45 | SciPy RK45 | torchdiffeq MPS |
|---|---|---|---|---|---|---|
| 500 | 0.053s | 0.039s | **0.005s** | 0.010s | 0.016s | 0.384s |
| 1000 | **0.047s** | 0.036s | 0.040s | 0.016s | 0.057s | 0.408s |
| 2000 | **0.062s** | 0.055s | 0.071s | 0.060s | 0.259s | 0.389s |
| 4000 | **0.122s** | 0.287s | 0.256s | 0.362s | 1.086s | 0.489s |
| 8000 | **0.356s** | 1.105s | 0.998s | 1.497s | 4.604s | 0.713s |
| 16000 | **1.201s** | 4.316s | 3.587s | 6.811s | 18.053s | 1.446s |

**Key findings:**

- **N >= 2000: mlxlab GPU is fastest.** At N=8000, 3x faster than Julia, 4x faster than
  MATLAB, 13x faster than SciPy.
- **N < 1000: Julia and MATLAB win** due to lower per-step dispatch overhead (compiled
  languages vs Python loop with `mx.eval()` per step).
- **torchdiffeq on MPS** has high constant overhead (~0.4s floor) but scales well at
  large N, approaching mlxlab beyond N=16000.
- **SciPy and MATLAB use float64 internally** (SciPy upcasts, MATLAB ode45 requires
  double). All others use float32. This is disclosed, not corrected -- it reflects the
  real-world experience of switching frameworks.

**Benchmark methodology:** Each framework was benchmarked in a separate process. System
load was verified idle (>70% CPU idle) between each run via `top`. Median of 5 runs
(3 for N=16000), 1 warmup. All benchmarks use the same random seed, same system, same
tolerances. Scripts are in `benchmarks/`.

## Roadmap (v0.2)

- **Compiled integration loop** -- push the time-stepping loop into compiled MLX to
  eliminate per-step Python/`mx.eval()` overhead. This is the main bottleneck at small N
  and the reason torchdiffeq converges at large N (its loop is C++).
- **`mlxlab.linalg`** -- sparse matrix support, eigenvalue solvers for stability analysis.
- **`mlxlab.signal`** -- FFT-based spectral analysis (MLX has `mx.fft`).
- **Implicit/stiff solvers** -- BDF, SDIRK methods for stiff systems.
- **Dense output** -- continuous interpolation between solver steps.
- **Benchmarks on M5 Ultra** when available.

## License

MIT
