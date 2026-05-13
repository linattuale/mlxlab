# mlxlab

Scientific computing library for Apple Silicon, built on MLX. Public repo:
`linattuale/mlxlab`.

## Scope

- `mlxlab.integrate` — ODE/SDE solvers
- `mlxlab.linalg` — det, slogdet, lstsq, matrix_rank, cond
- `mlxlab.signal` — fftfreq, rfftfreq, psd, welch, spectrogram
- `mlxlab.random` — exponential, gamma, beta, poisson, binomial

The flagship module is `mlxlab.integrate`. Prefer direct MLX arrays and GPU
acceleration where MLX supports it.

## Development

```bash
uv sync
PYTHONPATH=src uv run pytest tests/ -v
uv run ruff check .
```

Use the `src/` layout. Tests should run with `PYTHONPATH=src`.

## Notes

- `mlxlab.linalg` uses `stream=mx.cpu` because MLX 0.31 decompositions are CPU-only.
- `tests/test_linalg.py` uses subprocess isolation to work around an MLX 0.31 pytest crash with CPU-stream linalg ops.
- ODE solvers use `f(y, t)`, not SciPy's `f(t, y)`.
- Public API stays intentionally small; avoid thin wrappers unless they close a real MLX gap.
