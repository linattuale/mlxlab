# mlxlab

Scientific computing library for Apple Silicon, built on MLX.

## Overview

- **Current scope**:
  - `mlxlab.integrate` — ODE/SDE solvers
  - `mlxlab.linalg` — det, slogdet, lstsq, matrix_rank, cond
  - `mlxlab.signal` — fftfreq, rfftfreq, psd, welch, spectrogram
  - `mlxlab.random` — exponential, gamma, beta, poisson, binomial
- **Flagship module**: `mlxlab.integrate`
- **Key idea**: use MLX arrays directly, with GPU acceleration where MLX supports it

## Project Structure

```
src/mlxlab/
├── __init__.py
├── integrate/
│   ├── _api.py        # solve() entry point
│   ├── _base.py       # AbstractSolver
│   ├── _fixed.py      # Euler, RK4
│   ├── _adaptive.py   # Tsit5, Dopri5
│   ├── _stochastic.py # EulerMaruyama
│   ├── _stepsize.py   # Adaptive step-size controller
│   ├── _tableau.py    # Butcher tableaux
│   └── _solution.py   # Solution dataclass
├── linalg/
│   ├── __init__.py
│   └── _core.py       # det, slogdet, lstsq, matrix_rank, cond
├── signal/
│   ├── __init__.py
│   └── _core.py       # fftfreq, rfftfreq, psd, welch, spectrogram
└── random/
    ├── __init__.py
    └── _distributions.py
```

## Development

```bash
uv sync                    # install deps
PYTHONPATH=src uv run pytest tests/ -v
uv run ruff check .        # lint
```

## Notes

- `src/` layout is used throughout; tests should run with `PYTHONPATH=src`
- `mlxlab.linalg` uses `stream=mx.cpu` because MLX 0.31 decompositions are CPU-only
- `tests/test_linalg.py` uses subprocess isolation to work around an MLX 0.31 pytest crash with CPU-stream linalg ops

## Conventions

- src/ layout (standard for distributable packages)
- Build backend: hatchling
- Tests compare against SciPy/NumPy behavior where practical
- ODE solvers use `f(y, t)` rather than SciPy's `f(t, y)`
- Public API stays intentionally small; avoid adding thin wrappers unless they close a real MLX gap
