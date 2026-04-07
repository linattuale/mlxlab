"""Benchmark: mlxlab vs SciPy vs NumPy vs torchdiffeq (MPS) for ODE solving.

Solves dy/dt = (-y + tanh(W @ y + b)) / tau  (rate network)
at system sizes N = 100, 500, 1000, 2000.

MATLAB and Julia benchmarks are in separate scripts (bench_matlab.m, bench_julia.jl)
and their results are pasted into the final table.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

SIZES = [100, 500, 1000, 2000]
T_SPAN = (0.0, 1.0)
DT = 0.001
TAU = 0.01
N_WARMUP = 1
N_RUNS = 5


def make_system_np(N, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N, N)).astype(np.float32) * (0.5 / N**0.5)
    b = rng.normal(size=(N,)).astype(np.float32)
    y0 = rng.normal(size=(N,)).astype(np.float32) * 0.1
    return W, b, y0


def time_fn(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.std(times)


# --------------------------------------------------------------------------- #
# 1. mlxlab (MLX GPU)
# --------------------------------------------------------------------------- #

def bench_mlxlab(N):
    import mlx.core as mx
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import mlxlab as ml

    W_np, b_np, y0_np = make_system_np(N)
    W = mx.array(W_np)
    b = mx.array(b_np)
    y0 = mx.array(y0_np)

    def rhs(y, t):
        return (-y + mx.tanh(W @ y + b)) / TAU

    def run():
        sol = ml.integrate.solve(rhs, y0, t_span=T_SPAN, dt=DT, method="rk4")
        mx.eval(sol.y)
        return sol

    return time_fn(run)


# --------------------------------------------------------------------------- #
# 2. SciPy RK45 (CPU, adaptive)
# --------------------------------------------------------------------------- #

def bench_scipy(N):
    """SciPy doesn't have a built-in fixed-step RK4, so we use
    solve_ivp with DOP853 and dense output disabled, fixed max_step.
    Kept for reference but NOT included in the fair comparison table."""
    from scipy.integrate import solve_ivp

    W, b, y0 = make_system_np(N)

    def rhs(t, y):
        return (-y + np.tanh(W @ y + b)) / TAU

    def run():
        return solve_ivp(rhs, T_SPAN, y0, method="RK45", max_step=DT,
                         rtol=1e-4, atol=1e-6)

    return time_fn(run)


def bench_scipy_rk4(N):
    """Fair comparison: hand-rolled RK4 using SciPy/NumPy arrays.
    This is what a SciPy user would actually write for fixed-step RK4."""
    from scipy.linalg import blas

    W, b, y0 = make_system_np(N)

    def rhs(y):
        return (-y + np.tanh(W @ y + b)) / TAU

    def rk4_integrate():
        t = T_SPAN[0]
        y = y0.copy()
        dt = DT
        while t < T_SPAN[1]:
            h = min(dt, T_SPAN[1] - t)
            k1 = rhs(y)
            k2 = rhs(y + h / 2 * k1)
            k3 = rhs(y + h / 2 * k2)
            k4 = rhs(y + h * k3)
            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    return time_fn(rk4_integrate)


# --------------------------------------------------------------------------- #
# 3. NumPy hand-rolled RK4 (CPU)
# --------------------------------------------------------------------------- #

def bench_numpy_rk4(N):
    W, b, y0 = make_system_np(N)

    def rhs(y, t):
        return (-y + np.tanh(W @ y + b)) / TAU

    def rk4_integrate(f, y0, t0, t1, dt):
        t = t0
        y = y0.copy()
        while t < t1:
            h = min(dt, t1 - t)
            k1 = f(y, t)
            k2 = f(y + h / 2 * k1, t + h / 2)
            k3 = f(y + h / 2 * k2, t + h / 2)
            k4 = f(y + h * k3, t + h)
            y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    def run():
        return rk4_integrate(rhs, y0, T_SPAN[0], T_SPAN[1], DT)

    return time_fn(run)


# --------------------------------------------------------------------------- #
# 4. torchdiffeq + MPS (GPU, float32)
# --------------------------------------------------------------------------- #

def bench_torchdiffeq(N):
    import torch
    from torchdiffeq import odeint

    if not torch.backends.mps.is_available():
        return None, None

    W_np, b_np, y0_np = make_system_np(N)
    device = torch.device("mps")
    W = torch.tensor(W_np, device=device)
    b = torch.tensor(b_np, device=device)
    y0 = torch.tensor(y0_np, device=device)
    t_eval = torch.linspace(T_SPAN[0], T_SPAN[1], int((T_SPAN[1] - T_SPAN[0]) / DT) + 1,
                            device=device)

    class RHS(torch.nn.Module):
        def forward(self, t, y):
            return (-y + torch.tanh(W @ y + b)) / TAU

    func = RHS()

    def run():
        sol = odeint(func, y0, t_eval, method="rk4",
                     options={"dtype": torch.float32})
        torch.mps.synchronize()
        return sol

    return time_fn(run)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    results = {}

    for N in SIZES:
        print(f"\n{'='*60}")
        print(f"  N = {N}  (W is {N}x{N}, dt={DT}, T={T_SPAN[1]}s)")
        print(f"{'='*60}")

        row = {}

        # mlxlab
        try:
            med, std = bench_mlxlab(N)
            row["mlxlab_gpu"] = {"median": med, "std": std}
            print(f"  mlxlab (MLX GPU):     {med:.4f}s +/- {std:.4f}s")
        except Exception as e:
            print(f"  mlxlab: FAILED ({e})")

        # SciPy
        try:
            med, std = bench_scipy(N)
            row["scipy_cpu"] = {"median": med, "std": std}
            print(f"  SciPy RK45 (CPU):    {med:.4f}s +/- {std:.4f}s")
        except Exception as e:
            print(f"  SciPy: FAILED ({e})")

        # NumPy RK4
        try:
            med, std = bench_numpy_rk4(N)
            row["numpy_rk4_cpu"] = {"median": med, "std": std}
            print(f"  NumPy RK4 (CPU):     {med:.4f}s +/- {std:.4f}s")
        except Exception as e:
            print(f"  NumPy RK4: FAILED ({e})")

        # torchdiffeq
        try:
            med, std = bench_torchdiffeq(N)
            if med is not None:
                row["torchdiffeq_mps"] = {"median": med, "std": std}
                print(f"  torchdiffeq (MPS):   {med:.4f}s +/- {std:.4f}s")
            else:
                print(f"  torchdiffeq: MPS not available")
        except Exception as e:
            print(f"  torchdiffeq: FAILED ({e})")

        results[str(N)] = row

    # Save raw results
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY (median seconds, lower is better)")
    print(f"{'='*60}")
    header = f"{'N':>6} | {'mlxlab GPU':>12} | {'SciPy CPU':>12} | {'NumPy CPU':>12} | {'torchdiffeq':>12}"
    print(header)
    print("-" * len(header))
    for N in SIZES:
        row = results.get(str(N), {})
        vals = []
        for key in ["mlxlab_gpu", "scipy_cpu", "numpy_rk4_cpu", "torchdiffeq_mps"]:
            if key in row:
                vals.append(f"{row[key]['median']:.4f}s")
            else:
                vals.append("N/A")
        print(f"{N:>6} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12} | {vals[3]:>12}")


if __name__ == "__main__":
    main()
