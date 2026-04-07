"""Adaptive step-size control (PI controller)."""

import mlx.core as mx


def error_norm(error: mx.array) -> mx.array:
    """RMS norm of the error vector."""
    return mx.sqrt(mx.mean(error * error))


def propose_step(
    dt: mx.array,
    error: mx.array,
    order: int,
    atol: float,
    rtol: float,
    y: mx.array,
    safety: float = 0.9,
    min_factor: float = 0.2,
    max_factor: float = 10.0,
) -> tuple[bool, mx.array]:
    """Decide whether to accept and propose next step size.

    Uses standard error control from Hairer, Norsett & Wanner.

    Returns:
        (accept, dt_next)
    """
    scale = atol + rtol * mx.abs(y)
    err = error_norm(error / scale)
    # Avoid division by zero
    err = mx.maximum(err, mx.array(1e-10))

    exponent = 1.0 / (order + 1)
    factor = safety * (1.0 / err) ** exponent
    factor = float(mx.clip(factor, mx.array(min_factor), mx.array(max_factor)).item())

    accept = float(err.item()) <= 1.0
    dt_next = mx.array(float(dt.item()) * factor)
    return accept, dt_next
