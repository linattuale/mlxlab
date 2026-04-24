"""Main solve() entry point."""

from typing import Callable

import mlx.core as mx

from ._base import AbstractSolver
from ._fixed import Euler, RK4
from ._adaptive import Tsit5, Dopri5
from ._stochastic import EulerMaruyama
from ._stepsize import propose_step_array
from ._solution import Solution
from ._precision import wrap_rhs_dtype

# Unroll factor for compiled fixed-step chunks. 16 balances graph-trace amortization
# against compiled-graph size; smaller values leave dispatch overhead, larger values
# bloat the traced graph without measured benefit on M5.
_FIXED_CHUNK_SIZE = 16

_SOLVERS: dict[str, type[AbstractSolver]] = {
    "euler": Euler,
    "rk4": RK4,
    "tsit5": Tsit5,
    "dopri5": Dopri5,
    "euler_maruyama": EulerMaruyama,
}


def solve(
    f: Callable,
    y0: mx.array,
    t_span: tuple[float, float],
    *,
    method: str = "tsit5",
    dt: float | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-3,
    max_steps: int = 100_000,
    diffusion: Callable | None = None,
    saveat: mx.array | None = None,
    rhs_dtype=None,
) -> Solution:
    """Solve an ODE or SDE initial value problem.

    Args:
        f: Right-hand side function f(y, t) -> dy/dt.
        y0: Initial state (1-D mx.array).
        t_span: (t0, t1) integration interval. t1 must be > t0.
        method: Solver name — "euler", "rk4", "tsit5" (default), "dopri5", "euler_maruyama".
        dt: Fixed step size (must be positive). Required for fixed-step methods,
            optional for adaptive (used as initial step size).
        atol: Absolute tolerance (adaptive solvers only).
        rtol: Relative tolerance (adaptive solvers only).
        max_steps: Safety limit on number of steps.
        diffusion: Diffusion function g(y, t) for SDEs (euler_maruyama only).
        saveat: Optional sorted time points within [t0, t1] at which to save.
        rhs_dtype: Optional floating dtype for evaluating ``f``. When set, the
            solver state remains in ``y0.dtype`` but ``f`` sees ``y`` cast to
            ``rhs_dtype`` and its derivative is cast back to ``y0.dtype``.

    Returns:
        Solution with fields t, y, stats.
    """
    if method not in _SOLVERS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {list(_SOLVERS)}")

    solver = _SOLVERS[method]()
    t0, t1 = float(t_span[0]), float(t_span[1])

    if t1 <= t0:
        raise ValueError(f"t_span must satisfy t1 > t0, got ({t0}, {t1}).")
    if dt is not None and dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if saveat is not None:
        _validate_saveat(saveat, t0, t1)
    if rhs_dtype is not None:
        f = wrap_rhs_dtype(f, state_dtype=y0.dtype, rhs_dtype=rhs_dtype)

    if solver.is_adaptive:
        return _solve_adaptive(solver, f, y0, t0, t1, dt, atol, rtol, max_steps, saveat)
    elif isinstance(solver, EulerMaruyama):
        return _solve_sde(solver, f, y0, t0, t1, dt, diffusion, saveat)
    else:
        return _solve_fixed(solver, f, y0, t0, t1, dt, saveat)


def _validate_saveat(saveat: mx.array, t0: float, t1: float):
    """Check that saveat is sorted and within [t0, t1]."""
    sa = [float(saveat[i].item()) for i in range(saveat.shape[0])]
    for i in range(len(sa)):
        if sa[i] < t0 - 1e-12 or sa[i] > t1 + 1e-12:
            raise ValueError(
                f"saveat[{i}]={sa[i]} is outside t_span [{t0}, {t1}]."
            )
        if i > 0 and sa[i] < sa[i - 1] - 1e-12:
            raise ValueError(
                f"saveat must be sorted, but saveat[{i}]={sa[i]} < saveat[{i-1}]={sa[i-1]}."
            )


def _solve_fixed(solver, f, y0, t0, t1, dt, saveat):
    if dt is None:
        raise ValueError(f"{solver.__class__.__name__} requires dt.")

    n_full_steps = int((t1 - t0) / dt)
    last_dt = (t1 - t0) - n_full_steps * dt
    has_fractional_step = last_dt > 1e-12
    n_steps_total = n_full_steps
    if last_dt > 1e-12:
        n_steps_total += 1

    dt_arr = mx.array(dt)

    @mx.compile
    def compiled_chunk(y, t_arr):
        ys_chunk = []
        for _ in range(_FIXED_CHUNK_SIZE):
            y, _ = solver.step(f, t_arr, y, dt_arr)
            t_arr = t_arr + dt_arr
            ys_chunk.append(y)
        return y, mx.stack(ys_chunk)

    @mx.compile
    def compiled_step(y, t_arr):
        y_new, _ = solver.step(f, t_arr, y, dt_arr)
        return y_new

    t = t0
    y = y0
    ys_parts = [mx.expand_dims(y, axis=0)]

    n_chunks = n_full_steps // _FIXED_CHUNK_SIZE
    for _ in range(n_chunks):
        y, ys_chunk = compiled_chunk(y, mx.array(t))
        mx.eval(y, ys_chunk)
        t += _FIXED_CHUNK_SIZE * dt
        ys_parts.append(ys_chunk)

    for _ in range(n_full_steps % _FIXED_CHUNK_SIZE):
        y = compiled_step(y, mx.array(t))
        mx.eval(y)
        t += dt
        ys_parts.append(mx.expand_dims(y, axis=0))

    if has_fractional_step:
        y, _ = solver.step(f, mx.array(t), y, mx.array(last_dt))
        mx.eval(y)
        ys_parts.append(mx.expand_dims(y, axis=0))

    t_values = [t0 + i * dt for i in range(n_full_steps + 1)]
    if has_fractional_step:
        t_values.append(t1)

    sol = Solution(
        t=mx.array(t_values),
        y=mx.concatenate(ys_parts, axis=0),
        stats={"n_steps": n_steps_total},
    )

    if saveat is not None:
        sol = _interpolate_at(sol, saveat)

    return sol


def _solve_adaptive(solver, f, y0, t0, t1, dt, atol, rtol, max_steps, saveat):
    if dt is None:
        dt = (t1 - t0) * 1e-3

    t = t0
    y = y0
    current_dt = mx.array(dt)
    ts = [t]
    ys = [y]
    n_steps = 0
    n_rejected = 0

    @mx.compile
    def compiled_trial(y, t_arr, current_dt, remaining):
        step_dt = mx.minimum(current_dt, remaining)
        y_new, error = solver.step(f, t_arr, y, step_dt)
        accept, dt_next = propose_step_array(
            step_dt, error, solver.order, atol, rtol, y, y_new
        )
        y_out = mx.where(accept, y_new, y)
        dt_advance = mx.where(accept, step_dt, mx.array(0.0))
        return y_out, dt_next, dt_advance

    while t < t1 - 1e-12:
        if n_steps + n_rejected >= max_steps:
            raise RuntimeError(f"Max steps ({max_steps}) exceeded at t={t:.6g}.")

        y, dt_next, dt_advance = compiled_trial(
            y, mx.array(t), current_dt, mx.array(t1 - t)
        )
        mx.eval(y, dt_next, dt_advance)

        accepted_dt = float(dt_advance.item())
        if accepted_dt > 0.0:
            t = min(t + accepted_dt, t1)
            n_steps += 1
            ts.append(t)
            ys.append(y)
        else:
            n_rejected += 1

        current_dt = dt_next

    sol = Solution(
        t=mx.array(ts),
        y=mx.stack(ys),
        stats={"n_steps": n_steps, "n_accepted": n_steps, "n_rejected": n_rejected},
    )

    if saveat is not None:
        sol = _interpolate_at(sol, saveat)

    return sol


def _solve_sde(solver, f, y0, t0, t1, dt, diffusion, saveat):
    if dt is None:
        raise ValueError("EulerMaruyama requires dt.")

    n_full_steps = int((t1 - t0) / dt)
    last_dt = (t1 - t0) - n_full_steps * dt
    has_fractional_step = last_dt > 1e-12
    n_steps_total = n_full_steps
    if has_fractional_step:
        n_steps_total += 1

    dt_arr = mx.array(dt)

    t = t0
    y = y0
    ys_parts = [mx.expand_dims(y, axis=0)]

    if diffusion is None:
        @mx.compile
        def compiled_chunk(y, t_arr):
            ys_chunk = []
            for _ in range(_FIXED_CHUNK_SIZE):
                y, _ = solver.step(f, t_arr, y, dt_arr, diffusion=None)
                t_arr = t_arr + dt_arr
                ys_chunk.append(y)
            return y, mx.stack(ys_chunk)
    else:
        # Keep this Euler-Maruyama update in sync with EulerMaruyama.step in
        # _stochastic.py — the chunked path inlines the math for compilation
        # and will not pick up changes to solver.step.
        @mx.compile
        def compiled_chunk(y, t_arr, dws):
            ys_chunk = []
            for i in range(_FIXED_CHUNK_SIZE):
                drift = f(y, t_arr)
                noise = diffusion(y, t_arr)
                y = y + dt_arr * drift + noise * dws[i]
                t_arr = t_arr + dt_arr
                ys_chunk.append(y)
            return y, mx.stack(ys_chunk)

    n_chunks = n_full_steps // _FIXED_CHUNK_SIZE
    for _ in range(n_chunks):
        if diffusion is None:
            y, ys_chunk = compiled_chunk(y, mx.array(t))
        else:
            dws = mx.sqrt(dt_arr) * mx.random.normal((_FIXED_CHUNK_SIZE,) + y.shape)
            y, ys_chunk = compiled_chunk(y, mx.array(t), dws)
        mx.eval(y, ys_chunk)
        t += _FIXED_CHUNK_SIZE * dt
        ys_parts.append(ys_chunk)

    for _ in range(n_full_steps % _FIXED_CHUNK_SIZE):
        y, _ = solver.step(f, mx.array(t), y, dt_arr, diffusion=diffusion)
        mx.eval(y)
        t += dt
        ys_parts.append(mx.expand_dims(y, axis=0))

    if has_fractional_step:
        y, _ = solver.step(f, mx.array(t), y, mx.array(last_dt), diffusion=diffusion)
        mx.eval(y)
        ys_parts.append(mx.expand_dims(y, axis=0))

    t_values = [t0 + i * dt for i in range(n_full_steps + 1)]
    if has_fractional_step:
        t_values.append(t1)

    sol = Solution(
        t=mx.array(t_values),
        y=mx.concatenate(ys_parts, axis=0),
        stats={"n_steps": n_steps_total},
    )

    if saveat is not None:
        sol = _interpolate_at(sol, saveat)

    return sol


def _interpolate_at(sol: Solution, saveat: mx.array) -> Solution:
    """Linear interpolation of solution at requested time points.

    Assumes saveat is sorted and within the solution time range (validated upstream).
    """
    t_all = sol.t
    y_all = sol.y
    n_save = saveat.shape[0]

    counts = mx.sum(t_all[None, :] <= saveat[:, None], axis=1)
    idx = mx.clip(counts - 1, 0, t_all.shape[0] - 2).astype(mx.int32)

    t_lo = mx.take(t_all, idx)
    t_hi = mx.take(t_all, idx + 1)
    y_lo = mx.take(y_all, idx, axis=0)
    y_hi = mx.take(y_all, idx + 1, axis=0)

    denom = t_hi - t_lo
    alpha = mx.where(denom == 0, mx.zeros_like(denom), (saveat - t_lo) / denom)
    alpha = alpha.reshape((n_save,) + (1,) * (y_all.ndim - 1))
    ys_out = y_lo * (1 - alpha) + y_hi * alpha

    return Solution(
        t=saveat,
        y=ys_out,
        stats=sol.stats,
    )
