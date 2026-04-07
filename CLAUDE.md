# mlxlab

Scientific computing library for Apple Silicon, built on MLX.

## Overview

- **v0.1 scope**: `mlxlab.integrate` — ODE/SDE solvers
- **Solvers**: euler, rk4, tsit5 (default), dopri5, euler_maruyama
- **Key idea**: RHS evaluations (matrix-vector multiplies) run on GPU via MLX

## Project Structure

```
src/mlxlab/
├── __init__.py
└── integrate/
    ├── _api.py        # solve() entry point
    ├── _base.py       # AbstractSolver
    ├── _fixed.py      # Euler, RK4
    ├── _adaptive.py   # Tsit5, Dopri5
    ├── _stochastic.py # EulerMaruyama
    ├── _stepsize.py   # Adaptive step-size controller
    ├── _tableau.py    # Butcher tableaux
    └── _solution.py   # Solution dataclass
```

## Development

```bash
uv sync                    # install deps
uv run pytest tests/ -v    # run tests
uv run ruff check .        # lint
```

## Conventions

- src/ layout (standard for distributable packages)
- Build backend: hatchling
- Tests compare against SciPy reference solutions
- All solvers follow the same interface: `solver.step(f, t, y, dt) -> (y_next, error|None)`
- Public API through `mlxlab.integrate.solve()` — MATLAB-inspired
- Butcher tableaux stored as `mx.array` constants in `_tableau.py`
