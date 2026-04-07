"""Tests for mlxlab.random distributions.

Statistical tests: verify sample mean and variance against theoretical values.
Uses large sample sizes (10000+) for tight estimates.
"""

import mlx.core as mx
import numpy as np
import math
import pytest

import mlxlab as ml


def _check_moments(samples, expected_mean, expected_var, name, mean_tol=0.1, var_tol=0.2):
    """Check that sample mean and variance are close to expected."""
    mx.eval(samples)
    vals = np.array(samples)
    sample_mean = vals.mean()
    sample_var = vals.var()
    assert abs(sample_mean - expected_mean) < mean_tol * max(abs(expected_mean), 1.0), \
        f"{name} mean: {sample_mean:.4f} vs expected {expected_mean:.4f}"
    if expected_var > 0:
        assert abs(sample_var - expected_var) < var_tol * max(expected_var, 1.0), \
            f"{name} var: {sample_var:.4f} vs expected {expected_var:.4f}"


# ---- exponential -------------------------------------------------------------

def test_exponential_moments():
    mx.random.seed(42)
    scale = 2.0
    samples = ml.random.exponential(shape=(20000,), scale=scale)
    # E[X] = scale, Var[X] = scale^2
    _check_moments(samples, scale, scale**2, "exponential")


def test_exponential_positive():
    samples = ml.random.exponential(shape=(1000,))
    mx.eval(samples)
    assert mx.all(samples > 0).item(), "Exponential samples should be positive"


# ---- gamma -------------------------------------------------------------------

def test_gamma_shape_gt1():
    mx.random.seed(42)
    alpha, scale = 5.0, 2.0
    samples = ml.random.gamma(alpha, scale=scale, shape=(20000,))
    # E[X] = alpha*scale, Var[X] = alpha*scale^2
    _check_moments(samples, alpha * scale, alpha * scale**2, "gamma(5,2)")


def test_gamma_shape_lt1():
    mx.random.seed(42)
    alpha, scale = 0.5, 1.0
    samples = ml.random.gamma(alpha, scale=scale, shape=(20000,))
    _check_moments(samples, alpha * scale, alpha * scale**2, "gamma(0.5,1)",
                   mean_tol=0.15, var_tol=0.3)


def test_gamma_positive():
    mx.random.seed(42)
    samples = ml.random.gamma(2.0, shape=(1000,))
    mx.eval(samples)
    assert mx.all(samples > 0).item(), "Gamma samples should be positive"


# ---- beta --------------------------------------------------------------------

def test_beta_moments():
    mx.random.seed(42)
    a, b = 2.0, 5.0
    samples = ml.random.beta(a, b, shape=(20000,))
    expected_mean = a / (a + b)
    expected_var = (a * b) / ((a + b)**2 * (a + b + 1))
    _check_moments(samples, expected_mean, expected_var, "beta(2,5)")


def test_beta_range():
    mx.random.seed(42)
    samples = ml.random.beta(2.0, 3.0, shape=(1000,))
    mx.eval(samples)
    assert mx.all(samples >= 0).item() and mx.all(samples <= 1).item(), \
        "Beta samples should be in [0, 1]"


# ---- poisson -----------------------------------------------------------------

def test_poisson_small_lambda():
    mx.random.seed(42)
    lam = 3.0
    samples = ml.random.poisson(lam=lam, shape=(20000,))
    # E[X] = lam, Var[X] = lam
    _check_moments(samples, lam, lam, "poisson(3)")


def test_poisson_large_lambda():
    mx.random.seed(42)
    lam = 50.0
    samples = ml.random.poisson(lam=lam, shape=(20000,))
    _check_moments(samples, lam, lam, "poisson(50)", mean_tol=0.05, var_tol=0.15)


def test_poisson_nonnegative():
    mx.random.seed(42)
    samples = ml.random.poisson(lam=5.0, shape=(1000,))
    mx.eval(samples)
    assert mx.all(samples >= 0).item(), "Poisson samples should be non-negative"


# ---- binomial ----------------------------------------------------------------

def test_binomial_small():
    mx.random.seed(42)
    n, p = 20, 0.3
    samples = ml.random.binomial(n, p, shape=(20000,))
    # E[X] = np, Var[X] = np(1-p)
    _check_moments(samples, n * p, n * p * (1 - p), "binomial(20,0.3)")


def test_binomial_large():
    mx.random.seed(42)
    n, p = 200, 0.4
    samples = ml.random.binomial(n, p, shape=(20000,))
    _check_moments(samples, n * p, n * p * (1 - p), "binomial(200,0.4)",
                   mean_tol=0.05, var_tol=0.1)


def test_binomial_range():
    mx.random.seed(42)
    n = 10
    samples = ml.random.binomial(n, 0.5, shape=(1000,))
    mx.eval(samples)
    assert mx.all(samples >= 0).item() and mx.all(samples <= n).item(), \
        f"Binomial samples should be in [0, {n}]"
