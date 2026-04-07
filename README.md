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

print(sol.y[-1])  # ≈ exp(-5) = 0.006738
```

## Why mlxlab?

MLX gives you GPU-accelerated array operations on Apple Silicon. mlxlab builds
scientific computing tools on top — starting with ODE/SDE solvers where the
expensive right-hand side (matrix-vector multiplies) runs on the GPU.

```python
# 10,000-neuron rate network — the W @ r hits the GPU automatically
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

## License

MIT
