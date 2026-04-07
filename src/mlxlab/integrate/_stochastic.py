"""Stochastic solvers."""

import mlx.core as mx

from ._base import AbstractSolver


class EulerMaruyama(AbstractSolver):
    """Euler-Maruyama method for SDEs.

    Solves:  dy = f(y, t) dt + g(y, t) dW
    where dW ~ Normal(0, sqrt(dt)).
    """

    order = 1

    def step(self, f, t, y, dt, diffusion=None):
        y_next = y + dt * f(y, t)
        if diffusion is not None:
            dW = mx.sqrt(dt) * mx.random.normal(y.shape)
            y_next = y_next + diffusion(y, t) * dW
        return y_next, None
