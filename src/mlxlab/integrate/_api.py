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
        t_span: (t0, t1) integration interval.
        method: Solver name — "euler", "rk4", "tsit5" (default), "dopri5", "euler_maruyama".
        dt: Fixed step size. Required for fixed-step methods, optional for adaptive
            (used as initial step size).
        atol: Absolute tolerance (adaptive solvers only).
        rtol: Relative tolerance (adaptive solvers only).
        max_steps: Safety limit on number of steps.
        diffusion: Diffusion function g(y, t) for SDEs (euler_maruyama only).
        saveat: Optional time points at which to save the solution.

    Returns:
        Solution with fields t, y, stats.
    """
    if method not in _SOLVERS:
        raise ValueError(f"Unknown method {method!r}. Choose from: {list(_SOLVERS)}")

    solver = _SOLVERS[method]()
    t0, t1 = float(t_span[0]), float(t_span[1])

    if solver.is_adaptive:
        return _solve_adaptive(solver, f, y0, t0, t1, dt, atol, rtol, max_steps, saveat)
    elif isinstance(solver, EulerMaruyama):
        return _solve_sde(solver, f, y0, t0, t1, dt, diffusion)
    else:
        return _solve_fixed(solver, f, y0, t0, t1, dt, saveat)


def _solve_fixed(solver, f, y0, t0, t1, dt, saveat):
    if dt is None:
        raise ValueError(f"{solver.__class__.__name__} requires dt.")

    t = t0
    y = y0
    ts = [mx.array(t)]
    ys = [y]
    n_steps = 0

    while t < t1:
        step_dt = min(dt, t1 - t)
        y, _ = solver.step(f, mx.array(t), y, mx.array(step_dt))
        mx.eval(y)
        t += step_dt
        n_steps += 1
        ts.append(mx.array(t))
        ys.append(y)

    sol = Solution(
        t=mx.stack(ts),
        y=mx.stack(ys),
        stats={"n_steps": n_steps},
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

    while t < t1:
        if n_steps >= max_steps:
            raise RuntimeError(f"Max steps ({max_steps}) exceeded at t={t:.6g}.")

        # Don't overshoot
        step_dt = mx.minimum(current_dt, mx.array(t1 - t))

        y_new, error = solver.step(f, mx.array(t), y, step_dt)
        mx.eval(y_new, error)

        accept, dt_next = propose_step(step_dt, error, solver.order, atol, rtol, y_new)

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


def _solve_sde(solver, f, y0, t0, t1, dt, diffusion):
    if dt is None:
        raise ValueError("EulerMaruyama requires dt.")

    t = t0
    y = y0
    ts = [mx.array(t)]
    ys = [y]
    n_steps = 0

    while t < t1:
        step_dt = min(dt, t1 - t)
        y, _ = solver.step(f, mx.array(t), y, mx.array(step_dt), diffusion=diffusion)
        mx.eval(y)
        t += step_dt
        n_steps += 1
        ts.append(mx.array(t))
        ys.append(y)

    return Solution(
        t=mx.stack(ts),
        y=mx.stack(ys),
        stats={"n_steps": n_steps},
    )


def _interpolate_at(sol: Solution, saveat: mx.array) -> Solution:
    """Linear interpolation of solution at requested time points."""
    t_all = sol.t
    y_all = sol.y
    n_save = saveat.shape[0]
    n_dims = y_all.shape[1] if y_all.ndim > 1 else 1

    ys_out = []
    j = 0
    for i in range(n_save):
        ti = float(saveat[i].item())
        # Advance j until t_all[j] >= ti
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
