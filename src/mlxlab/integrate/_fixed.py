"""Fixed-step solvers: Euler, RK4."""

import mlx.core as mx

from ._base import AbstractSolver


class Euler(AbstractSolver):
    """Forward Euler method (1st order)."""

    order = 1

    def step(self, f, t, y, dt):
        return y + dt * f(y, t), None


class RK4(AbstractSolver):
    """Classic 4th-order Runge-Kutta."""

    order = 4

    def step(self, f, t, y, dt):
        half_dt = dt / 2
        k1 = f(y, t)
        k2 = f(y + half_dt * k1, t + half_dt)
        k3 = f(y + half_dt * k2, t + half_dt)
        k4 = f(y + dt * k3, t + dt)
        return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4), None
