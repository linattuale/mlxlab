"""Solution container."""

from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class Solution:
    """Result of an ODE/SDE integration.

    Attributes:
        t: Time points, shape (n_steps,).
        y: State trajectory, shape (n_steps, *state_shape).
        stats: Solver statistics (n_steps, n_accepted, n_rejected).
    """

    t: mx.array
    y: mx.array
    stats: dict = field(default_factory=dict)
