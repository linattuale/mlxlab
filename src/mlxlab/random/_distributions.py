"""Random distributions built on MLX primitives.

Algorithms:
- exponential: inverse CDF of uniform
- gamma: Marsaglia & Tsang (2000) for shape >= 1, Ahrens-Dieter boost for shape < 1
- beta: from two independent gammas
- poisson: inverse CDF for small lambda, normal approximation for large lambda
- binomial: inversion for small n*p, normal approximation for large n*p

Key handling: when an explicit key is passed, each internal random draw uses a
fresh subkey via mx.random.split to ensure independence.
"""

import mlx.core as mx
import math


def _next_key(key):
    """Split a key and return (subkey_for_use, next_key). If key is None, return (None, None)."""
    if key is None:
        return None, None
    pair = mx.random.split(key)
    return pair[0], pair[1]


def exponential(
    shape: tuple[int, ...] = (),
    scale: float = 1.0,
    key: mx.array | None = None,
) -> mx.array:
    """Exponential distribution via inverse CDF.

    Args:
        shape: Output shape.
        scale: Scale parameter (1/rate, must be > 0). Mean of the distribution.
        key: Optional PRNG key.

    Returns:
        Samples from Exp(1/scale).
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    subkey, key = _next_key(key)
    u = mx.random.uniform(shape=shape, key=subkey)
    u = mx.clip(u, 1e-7, 1.0)
    return -scale * mx.log(u)


def gamma(
    shape_param: float,
    scale: float = 1.0,
    shape: tuple[int, ...] = (),
    key: mx.array | None = None,
) -> mx.array:
    """Gamma distribution via Marsaglia & Tsang's method.

    Args:
        shape_param: Shape parameter (alpha > 0).
        scale: Scale parameter (must be > 0).
        shape: Output shape.
        key: Optional PRNG key.

    Returns:
        Samples from Gamma(alpha, scale).
    """
    if shape_param <= 0:
        raise ValueError(f"shape_param must be > 0, got {shape_param}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    if shape_param < 1.0:
        boost = True
        alpha = shape_param + 1.0
    else:
        boost = False
        alpha = shape_param

    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)

    samples = mx.zeros(shape)
    remaining = mx.ones(shape, dtype=mx.bool_)
    max_iter = 200

    for _ in range(max_iter):
        if not mx.any(remaining).item():
            break

        subkey, key = _next_key(key)
        z = mx.random.normal(shape=shape, key=subkey)
        subkey, key = _next_key(key)
        u = mx.random.uniform(shape=shape, key=subkey)

        v = (1.0 + c * z) ** 3
        valid_v = v > 0
        log_accept = 0.5 * z * z + d - d * v + d * mx.log(v)
        accepted = valid_v & (mx.log(u) < log_accept) & remaining

        samples = mx.where(accepted, d * v, samples)
        remaining = remaining & ~accepted

    if boost:
        subkey, key = _next_key(key)
        u_boost = mx.random.uniform(shape=shape, key=subkey)
        u_boost = mx.clip(u_boost, 1e-7, 1.0)
        samples = samples * (u_boost ** (1.0 / shape_param))

    return samples * scale


def beta(
    a: float,
    b: float,
    shape: tuple[int, ...] = (),
    key: mx.array | None = None,
) -> mx.array:
    """Beta distribution from two independent gammas.

    Args:
        a: Shape parameter alpha > 0.
        b: Shape parameter beta > 0.
        shape: Output shape.
        key: Optional PRNG key.

    Returns:
        Samples from Beta(a, b) in [0, 1].
    """
    # Split key so x and y get independent streams
    if key is not None:
        pair = mx.random.split(key)
        key_x, key_y = pair[0], pair[1]
    else:
        key_x, key_y = None, None

    x = gamma(a, shape=shape, key=key_x)
    y = gamma(b, shape=shape, key=key_y)
    return x / (x + y)


def poisson(
    lam: float = 1.0,
    shape: tuple[int, ...] = (),
    key: mx.array | None = None,
) -> mx.array:
    """Poisson distribution.

    Uses inverse CDF for small lambda, normal approximation for large lambda.

    Args:
        lam: Rate parameter (lambda >= 0).
        shape: Output shape.
        key: Optional PRNG key.

    Returns:
        Samples from Poisson(lambda), dtype int32.
    """
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if lam == 0:
        return mx.zeros(shape, dtype=mx.int32)

    if lam < 30:
        L = math.exp(-lam)
        p = mx.ones(shape)
        k = mx.zeros(shape)

        max_iter = int(lam * 4) + 40
        for _ in range(max_iter):
            subkey, key = _next_key(key)
            u = mx.random.uniform(shape=shape, key=subkey)
            p = p * u
            still_going = p > L
            k = k + mx.where(still_going, mx.array(1.0), mx.array(0.0))
            if not mx.any(still_going).item():
                break

        return k.astype(mx.int32)
    else:
        subkey, key = _next_key(key)
        z = mx.random.normal(shape=shape, key=subkey)
        samples = lam + math.sqrt(lam) * z
        return mx.maximum(mx.floor(samples + 0.5), mx.array(0.0)).astype(mx.int32)


def binomial(
    n: int,
    p: float,
    shape: tuple[int, ...] = (),
    key: mx.array | None = None,
) -> mx.array:
    """Binomial distribution.

    Uses direct summation for small n, normal approximation for large n.

    Args:
        n: Number of trials (>= 0).
        p: Success probability per trial, in [0, 1].
        shape: Output shape.
        key: Optional PRNG key.

    Returns:
        Samples from Binomial(n, p), dtype int32.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")

    if n * p < 20 and n < 100:
        count = mx.zeros(shape)
        for _ in range(n):
            subkey, key = _next_key(key)
            u = mx.random.uniform(shape=shape, key=subkey)
            count = count + mx.where(u < p, mx.array(1.0), mx.array(0.0))
        return count.astype(mx.int32)
    else:
        mu = n * p
        sigma = math.sqrt(n * p * (1 - p))
        subkey, key = _next_key(key)
        z = mx.random.normal(shape=shape, key=subkey)
        # Continuity correction
        samples = mu + sigma * z + 0.5
        return mx.clip(mx.floor(samples), 0.0, float(n)).astype(mx.int32)
