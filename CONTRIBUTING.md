# Contributing to mlxlab

Thanks for your interest in improving `mlxlab`.

## Scope

`mlxlab` is an alpha scientific computing library built for Apple Silicon and MLX.
The project is intentionally narrow: we would rather ship a small number of
correct, well-documented tools than a wide surface area of thin wrappers.

## Development Setup

```bash
uv sync --dev
```

## Running Checks

```bash
PYTHONPATH=src uv run pytest tests/ -v
uv run ruff check .
```

Notes:

- The project uses a `src/` layout, so tests should run with `PYTHONPATH=src`.
- `mlxlab.linalg` uses MLX CPU-stream decompositions in 0.31. Those tests run via
  subprocess isolation because MLX currently crashes when some CPU-stream linalg
  ops are exercised directly under `pytest`.

## What Makes a Good Contribution

- Fixes a correctness bug, edge case, or documentation gap.
- Adds a small, focused API that MLX does not already provide well.
- Includes tests for the new behavior.
- Keeps the README honest about limitations and maturity.

For numerical changes, please include one or more of:

- a paper or algorithm reference
- a comparison against NumPy/SciPy/Julia/Matlab behavior
- a short explanation of tradeoffs or approximations

## Benchmarks

Performance claims should be reproducible and checked into `benchmarks/`.
If a change affects `mlxlab.integrate`, benchmark evidence is strongly preferred.
For other modules, lightweight benchmark additions are welcome when they support a
clear story for users.

## Pull Requests

Small, focused pull requests are easiest to review.

Please include:

- what changed
- why it changed
- how you validated it
- any limitations or follow-up work

If the change affects public API or numerical behavior, update `README.md` and
`CHANGELOG.md` in the same pull request.
