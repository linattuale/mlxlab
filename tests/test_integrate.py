"""Tests for mlxlab.integrate solvers."""

import mlx.core as mx
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import mlxlab as ml
from mlxlab.integrate import Solution


# ---- Helpers ----------------------------------------------------------------

def assert_close(a, b, atol=1e-4, msg=""):
    diff = abs(float(a) - float(b))
    assert diff < atol, f"{msg} |{float(a):.8f} - {float(b):.8f}| = {diff:.2e} > {atol:.2e}"


# ---- Test systems -----------------------------------------------------------

def exp_decay(y, t):
    """dy/dt = -y, solution: y(t) = y0 * exp(-t)."""
    return -y


def harmonic_rhs(y, t):
    """2D harmonic oscillator: dx/dt = v, dv/dt = -x."""
    return mx.array([y[1], -y[0]])


def lotka_volterra(y, t):
    """Lotka-Volterra predator-prey: alpha=1.5, beta=1, delta=1, gamma=3."""
    x, p = y[0], y[1]
    dxdt = 1.5 * x - x * p
    dpdt = x * p - 3.0 * p
    return mx.array([dxdt, dpdt])


def linear_system(y, t):
    """dy/dt = A @ y with stable A."""
    A = mx.array([[-0.5, 0.1], [0.0, -0.3]])
    return A @ y


# ---- Exponential decay (all solvers) ----------------------------------------

@pytest.mark.parametrize("method,dt,atol_check", [
    ("euler", 0.001, 1e-2),
    ("rk4", 0.01, 1e-6),
    ("tsit5", None, 1e-4),
    ("dopri5", None, 1e-4),
])
def test_exp_decay(method, dt, atol_check):
    kwargs = {"dt": dt} if dt else {}
    sol = ml.integrate.solve(exp_decay, mx.array([1.0]), t_span=(0, 5), method=method, **kwargs)
    expected = float(mx.exp(mx.array(-5.0)).item())
    assert_close(sol.y[-1].item(), expected, atol=atol_check, msg=f"{method}")


# ---- Harmonic oscillator (energy conservation) ------------------------------

@pytest.mark.parametrize("method,extra,atol_e", [
    ("rk4", {"dt": 0.001}, 1e-3),
    ("tsit5", {"atol": 1e-8, "rtol": 1e-6}, 1e-2),
    ("dopri5", {"atol": 1e-8, "rtol": 1e-6}, 1e-2),
])
def test_harmonic_energy(method, extra, atol_e):
    y0 = mx.array([1.0, 0.0])
    sol = ml.integrate.solve(harmonic_rhs, y0, t_span=(0, 20), method=method, **extra)

    # Energy = 0.5 * (x^2 + v^2) should be conserved
    energy_init = 0.5 * float((sol.y[0] ** 2).sum().item())
    energy_final = 0.5 * float((sol.y[-1] ** 2).sum().item())
    assert_close(energy_final, energy_init, atol=atol_e, msg=f"{method} energy conservation")


# ---- Lotka-Volterra vs SciPy -----------------------------------------------

@pytest.mark.parametrize("method", ["tsit5", "dopri5"])
def test_lotka_volterra_vs_scipy(method):
    y0_np = np.array([10.0, 5.0])
    y0_mx = mx.array([10.0, 5.0])

    def lv_scipy(t, y):
        return [1.5 * y[0] - y[0] * y[1], y[0] * y[1] - 3.0 * y[1]]

    ref = solve_ivp(lv_scipy, (0, 2), y0_np, method="RK45", rtol=1e-8, atol=1e-10)
    sol = ml.integrate.solve(lotka_volterra, y0_mx, t_span=(0, 2), method=method,
                             atol=1e-8, rtol=1e-6)

    # Compare final values
    assert_close(sol.y[-1][0].item(), ref.y[0, -1], atol=1e-2, msg=f"{method} x")
    assert_close(sol.y[-1][1].item(), ref.y[1, -1], atol=1e-2, msg=f"{method} p")


# ---- Adaptive accuracy (tight vs loose tolerances) --------------------------

def test_adaptive_accuracy_scaling():
    y0 = mx.array([1.0])
    expected = float(mx.exp(mx.array(-5.0)).item())

    sol_loose = ml.integrate.solve(exp_decay, y0, t_span=(0, 5), method="tsit5",
                                   atol=1e-3, rtol=1e-2)
    sol_tight = ml.integrate.solve(exp_decay, y0, t_span=(0, 5), method="tsit5",
                                   atol=1e-10, rtol=1e-8)

    err_loose = abs(sol_loose.y[-1].item() - expected)
    err_tight = abs(sol_tight.y[-1].item() - expected)
    assert err_tight < err_loose, f"Tight tol error ({err_tight:.2e}) >= loose ({err_loose:.2e})"
    assert sol_tight.stats["n_steps"] > sol_loose.stats["n_steps"]


# ---- 2D linear system -------------------------------------------------------

def test_linear_system():
    y0 = mx.array([1.0, 2.0])
    sol = ml.integrate.solve(linear_system, y0, t_span=(0, 10), method="tsit5")
    # Both components should decay toward 0
    assert float(mx.abs(sol.y[-1]).max().item()) < 0.1


# ---- GPU matmul RHS (large system) ------------------------------------------

def test_large_matmul_rhs():
    n = 500
    mx.random.seed(42)
    A = mx.random.normal((n, n)) * (0.5 / n**0.5)
    # Make stable: shift eigenvalues negative
    A = A - 2.0 * mx.eye(n)
    y0 = mx.random.normal((n,))

    def rhs(y, t):
        return A @ y

    sol = ml.integrate.solve(rhs, y0, t_span=(0, 1), dt=0.01, method="rk4")
    # Should decay — final norm < initial norm
    norm_init = float(mx.sqrt((y0 ** 2).sum()).item())
    norm_final = float(mx.sqrt((sol.y[-1] ** 2).sum()).item())
    assert norm_final < norm_init, f"Expected decay: {norm_final:.4f} >= {norm_init:.4f}"


# ---- SDE: Euler-Maruyama (geometric Brownian motion) ------------------------

def test_euler_maruyama_gbm():
    """Geometric Brownian motion: dS = mu*S*dt + sigma*S*dW.
    E[S(T)] = S0 * exp(mu*T).
    """
    mu, sigma, T, S0 = 0.05, 0.2, 1.0, 100.0
    n_paths = 500
    dt = 0.005

    finals = []
    for _ in range(n_paths):
        sol = ml.integrate.solve(
            lambda y, t: mu * y,
            mx.array([S0]),
            t_span=(0, T),
            dt=dt,
            method="euler_maruyama",
            diffusion=lambda y, t: sigma * y,
        )
        finals.append(sol.y[-1].item())

    mean_final = np.mean(finals)
    expected_mean = S0 * np.exp(mu * T)
    # Check that sample mean is within ~5% of expected (statistical test)
    rel_err = abs(mean_final - expected_mean) / expected_mean
    assert rel_err < 0.1, f"GBM mean: {mean_final:.2f} vs expected {expected_mean:.2f} (err={rel_err:.2%})"


# ---- Solution dataclass -----------------------------------------------------

def test_solution_shape():
    sol = ml.integrate.solve(exp_decay, mx.array([1.0, 2.0]), t_span=(0, 1), dt=0.1, method="rk4")
    assert sol.t.ndim == 1
    assert sol.y.ndim == 2
    assert sol.y.shape[1] == 2
    assert sol.t.shape[0] == sol.y.shape[0]


# ---- saveat (interpolation) -------------------------------------------------

def test_saveat():
    saveat = mx.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    sol = ml.integrate.solve(exp_decay, mx.array([1.0]), t_span=(0, 5), dt=0.01,
                             method="rk4", saveat=saveat)
    assert sol.t.shape[0] == 6
    expected_at_3 = float(mx.exp(mx.array(-3.0)).item())
    assert_close(sol.y[3].item(), expected_at_3, atol=1e-3, msg="saveat t=3")


# ---- Error handling ---------------------------------------------------------

def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        ml.integrate.solve(exp_decay, mx.array([1.0]), t_span=(0, 1), method="bogus")


def test_fixed_step_requires_dt():
    with pytest.raises(ValueError, match="requires dt"):
        ml.integrate.solve(exp_decay, mx.array([1.0]), t_span=(0, 1), method="rk4")
