"""Main solve() entry point."""

from typing import Callable

import mlx.core as mx

from ._base import AbstractSolver
from ._fixed import Euler, RK4
from ._adaptive import Tsit5, Dopri5
from ._stochastic import EulerMaruyama
from ._stepsize import propose_step
from ._solution import Solution

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

    n_steps_total = int((t1 - t0) / dt)
    last_dt = (t1 - t0) - n_steps_total * dt
    if last_dt > 1e-12:
        n_steps_total += 1

    dt_arr = mx.array(dt)

    # Compile a single step: eliminates graph tracing overhead per iteration
    @mx.compile
    def compiled_step(y, t_arr):
        y_new, _ = solver.step(f, t_arr, y, dt_arr)
        return y_new

    t = t0
    y = y0
    ts = [mx.array(t)]
    ys = [y]

    for i in range(n_steps_total):
        step_dt = min(dt, t1 - t)
        if abs(step_dt - dt) > 1e-12:
            # Last fractional step — can't use compiled version (different dt)
            y, _ = solver.step(f, mx.array(t), y, mx.array(step_dt))
        else:
            y = compiled_step(y, mx.array(t))

        mx.eval(y)
        t += step_dt
        ts.append(mx.array(t))
        ys.append(y)

    sol = Solution(
        t=mx.stack(ts),
        y=mx.stack(ys),
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
    ts = [mx.array(t)]
    ys = [y]
    n_steps = 0
    n_rejected = 0

    while t < t1 - 1e-12:
        if n_steps + n_rejected >= max_steps:
            raise RuntimeError(f"Max steps ({max_steps}) exceeded at t={t:.6g}.")

        # Don't overshoot
        step_dt = mx.minimum(current_dt, mx.array(t1 - t))

        y_new, error = solver.step(f, mx.array(t), y, step_dt)
        mx.eval(y_new, error)

        accept, dt_next = propose_step(step_dt, error, solver.order, atol, rtol, y, y_new)

        if accept:
            t += float(step_dt.item())
            y = y_new
            n_steps += 1
            ts.append(mx.array(t))
            ys.append(y)
        else:
            n_rejected += 1

        current_dt = dt_next

    sol = Solution(
        t=mx.stack(ts),
        y=mx.stack(ys),
        stats={"n_steps": n_steps, "n_accepted": n_steps, "n_rejected": n_rejected},
    )

    if saveat is not None:
        sol = _interpolate_at(sol, saveat)

    return sol


def _solve_sde(solver, f, y0, t0, t1, dt, diffusion, saveat):
    if dt is None:
        raise ValueError("EulerMaruyama requires dt.")

    n_steps_total = round((t1 - t0) / dt)
    actual_dt = (t1 - t0) / n_steps_total  # avoid float accumulation drift

    t = t0
    y = y0
    ts = [mx.array(t)]
    ys = [y]

    for i in range(n_steps_total):
        y, _ = solver.step(f, mx.array(t), y, mx.array(actual_dt), diffusion=diffusion)
        mx.eval(y)
        t = t0 + (i + 1) * actual_dt  # exact time, no accumulation
        ts.append(mx.array(t))
        ys.append(y)

    sol = Solution(
        t=mx.stack(ts),
        y=mx.stack(ys),
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

    ys_out = []
    j = 0
    for i in range(n_save):
        ti = float(saveat[i].item())
        # Advance j until t_all[j+1] >= ti
        while j < t_all.shape[0] - 1 and float(t_all[j + 1].item()) < ti:
            j += 1
        if j >= t_all.shape[0] - 1:
            ys_out.append(y_all[-1])
        else:
            t_lo = float(t_all[j].item())
            t_hi = float(t_all[j + 1].item())
            if t_hi == t_lo:
                ys_out.append(y_all[j])
            else:
                alpha = (ti - t_lo) / (t_hi - t_lo)
                ys_out.append(y_all[j] * (1 - alpha) + y_all[j + 1] * alpha)

    return Solution(
        t=saveat,
        y=mx.stack(ys_out),
        stats=sol.stats,
    )
