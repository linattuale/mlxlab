"""Additional random distributions built on mlx.core.random.

MLX ships: uniform, normal, multivariate_normal, bernoulli, categorical,
truncated_normal, gumbel, laplace, randint, permutation.

mlxlab.random adds: exponential, gamma, beta, poisson, binomial.

For the base distributions, use mlx.core.random directly.
"""

from ._distributions import exponential, gamma, beta, poisson, binomial

__all__ = ["exponential", "gamma", "beta", "poisson", "binomial"]
