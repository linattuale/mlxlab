"""Adaptive Runge-Kutta solvers: Tsit5, Dopri5."""

import mlx.core as mx

from ._base import AbstractSolver
from ._tableau import (
    TSIT5_A, TSIT5_B, TSIT5_C, TSIT5_E,
    DOPRI5_A, DOPRI5_B, DOPRI5_C, DOPRI5_E,
)


class _EmbeddedRK(AbstractSolver):
    """Generic embedded Runge-Kutta solver (FSAL)."""

    order: int
    _c: list
    _a: list
    _b: mx.array
    _e: mx.array
    n_stages: int = 7

    @property
    def is_adaptive(self) -> bool:
        return True

    def step(self, f, t, y, dt):
        c, a, b, e = self._c, self._a, self._b, self._e

        # Compute stages
        k = [None] * self.n_stages
        k[0] = f(y, t)
        for i in range(1, self.n_stages):
            yi = y
            for j in range(i):
                yi = yi + dt * a[i][j] * k[j]
            k[i] = f(yi, t + c[i] * dt)

        # 5th order solution
        y_next = y
        for i in range(self.n_stages):
            y_next = y_next + dt * b[i] * k[i]

        # Error estimate: sum of e[i] * k[i] * dt
        error = mx.zeros_like(y)
        for i in range(self.n_stages):
            error = error + dt * e[i] * k[i]

        return y_next, error


class Tsit5(_EmbeddedRK):
    """Tsitouras 5(4) — default adaptive solver.

    More efficient than Dormand-Prince for most non-stiff problems.
    Ref: Tsitouras (2011), Computers & Math. with Applications 62, 770-775.
    """

    order = 5
    _c = TSIT5_C
    _a = TSIT5_A
    _b = TSIT5_B
    _e = TSIT5_E


class Dopri5(_EmbeddedRK):
    """Dormand-Prince 5(4) — the classic ode45 method."""

    order = 5
    _c = DOPRI5_C
    _a = DOPRI5_A
    _b = DOPRI5_B
    _e = DOPRI5_E
