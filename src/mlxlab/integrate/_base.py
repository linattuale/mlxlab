"""Base class for all solvers."""

from abc import ABC, abstractmethod

import mlx.core as mx


class AbstractSolver(ABC):
    """Base class for ODE/SDE solvers."""

    order: int

    @abstractmethod
    def step(self, f, t: mx.array, y: mx.array, dt: mx.array) -> tuple[mx.array, mx.array | None]:
        """Advance one step.

        Returns:
            (y_next, error_estimate) where error_estimate is None for fixed-step solvers.
        """
        ...

    @property
    def is_adaptive(self) -> bool:
        return False
