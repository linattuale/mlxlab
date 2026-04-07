"""Random distributions built on MLX primitives.

Algorithms:
- exponential: inverse CDF of uniform
- gamma: Marsaglia & Tsang (2000) for shape >= 1, Ahrens-Dieter boost for shape < 1
- beta: from two independent gammas
- poisson: inverse CDF for small lambda, normal approximation for large lambda
- binomial: inversion for small n*p, normal approximation for large n*p
"""

import mlx.core as mx
import math


def exponential(
    shape: tuple[int, ...] = (),
    scale: float = 1.0,
    key: mx.array | None = None,
) -> mx.array:
    """Exponential distribution via inverse CDF.

    Args:
        shape: Output shape.
        scale: Scale parameter (1/rate). Mean of the distribution.
        key: Optional PRNG key.

    Returns:
        Samples from Exp(1/scale).
    """
    u = mx.random.uniform(shape=shape, key=key)
    # Clamp away from 0 to avoid log(0)
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
        scale: Scale parameter (beta > 0).
        shape: Output shape.
        key: Optional PRNG key.

    Returns:
        Samples from Gamma(alpha, beta).
    """
    if shape_param <= 0:
        raise ValueError(f"shape_param must be > 0, got {shape_param}")

    # For alpha < 1, use boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
    if shape_param < 1.0:
        boost = True
        alpha = shape_param + 1.0
    else:
        boost = False
        alpha = shape_param

    # Marsaglia & Tsang (2000): "A Simple Method for Generating Gamma Variables"
    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)

    # Rejection sampling
    samples = mx.zeros(shape)
    remaining = mx.ones(shape, dtype=mx.bool_)
    max_iter = 200

    for _ in range(max_iter):
        if not mx.any(remaining).item():
            break

        z = mx.random.normal(shape=shape, key=key)
        v = (1.0 + c * z) ** 3
        u = mx.random.uniform(shape=shape, key=key)

        # Accept conditions
        valid_v = v > 0
        log_accept = 0.5 * z * z + d - d * v + d * mx.log(v)
        accepted = valid_v & (mx.log(u) < log_accept) & remaining

        samples = mx.where(accepted, d * v, samples)
        remaining = remaining & ~accepted

    # Boost for alpha < 1
    if boost:
        u_boost = mx.random.uniform(shape=shape, key=key)
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

    Beta(a, b) = X / (X + Y) where X ~ Gamma(a), Y ~ Gamma(b).

    Args:
        a: Shape parameter alpha > 0.
        b: Shape parameter beta > 0.
        shape: Output shape.
        key: Optional PRNG key.

    Returns:
        Samples from Beta(a, b) in [0, 1].
    """
    x = gamma(a, shape=shape, key=key)
    y = gamma(b, shape=shape, key=key)
    return x / (x + y)


def poisson(
    lam: float = 1.0,
    shape: tuple[int, ...] = (),
    key: mx.array | None = None,
) -> mx.array:
    """Poisson distribution.

    Uses inverse CDF for small lambda, normal approximation for large lambda.

    Args:
        lam: Rate parameter (lambda > 0).
        shape: Output shape.
        key: Optional PRNG key.

    Returns:
        Integer samples from Poisson(lambda).
    """
    if lam <= 0:
        raise ValueError(f"lam must be > 0, got {lam}")

    if lam < 30:
        # Inverse CDF method (Knuth)
        L = math.exp(-lam)
        result = mx.zeros(shape)
        p = mx.ones(shape)
        k = mx.zeros(shape)

        max_iter = int(lam * 4) + 40
        for _ in range(max_iter):
            u = mx.random.uniform(shape=shape, key=key)
            p = p * u
            still_going = p > L
            k = k + mx.where(still_going, mx.array(1.0), mx.array(0.0))
            if not mx.any(still_going).item():
                break

        return k
    else:
        # Normal approximation: Poisson(lam) ~ Normal(lam, sqrt(lam))
        z = mx.random.normal(shape=shape, key=key)
        samples = lam + math.sqrt(lam) * z
        # Round to nearest non-negative integer
        return mx.maximum(mx.floor(samples + 0.5), mx.array(0.0))


def binomial(
    n: int,
    p: float,
    shape: tuple[int, ...] = (),
    key: mx.array | None = None,
) -> mx.array:
    """Binomial distribution.

    Uses direct summation for small n, normal approximation for large n.

    Args:
        n: Number of trials.
        p: Success probability per trial.
        shape: Output shape.
        key: Optional PRNG key.

    Returns:
        Integer samples from Binomial(n, p).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")

    if n * p < 20 and n < 100:
        # Direct: sum of n Bernoulli trials
        count = mx.zeros(shape)
        for _ in range(n):
            u = mx.random.uniform(shape=shape, key=key)
            count = count + mx.where(u < p, mx.array(1.0), mx.array(0.0))
        return count
    else:
        # Normal approximation: Bin(n,p) ~ Normal(np, sqrt(np(1-p)))
        mu = n * p
        sigma = math.sqrt(n * p * (1 - p))
        z = mx.random.normal(shape=shape, key=key)
        samples = mu + sigma * z
        return mx.clip(mx.floor(samples + 0.5), 0.0, float(n))
