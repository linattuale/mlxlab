"""Benchmark: Dormand-Prince 5(4) across all frameworks.

Same algorithm, same tolerances, same problem.
Solves dy/dt = (-y + tanh(W @ y + b)) / tau  (rate network)

Usage:
    uv run python bench_dp5.py mlxlab
    uv run python bench_dp5.py scipy
    uv run python bench_dp5.py numpy
    uv run python bench_dp5.py torchdiffeq
    uv run python bench_dp5.py all
    uv run python bench_dp5.py plot
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

SIZES = [500, 1000, 2000, 4000, 8000]
T_SPAN = (0.0, 1.0)
TAU = 0.01
GAIN = 1.5  # chaotic regime (>1)
RTOL = 1e-4
ATOL = 1e-6
N_WARMUP = 1
N_RUNS = 5
RESULTS_FILE = Path(__file__).parent / "results_dp5.json"


def make_system_np(N, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N, N)).astype(np.float32) * (GAIN / N**0.5)
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
# mlxlab dopri5 (MLX GPU)
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
        sol = ml.integrate.solve(rhs, y0, t_span=T_SPAN, method="dopri5",
                                 atol=ATOL, rtol=RTOL)
        mx.eval(sol.y)
        return sol

    return time_fn(run)


# --------------------------------------------------------------------------- #
# mlxlab dopri5 (MLX CPU)
# --------------------------------------------------------------------------- #

def bench_mlxlab_cpu(N):
    import mlx.core as mx
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import mlxlab as ml

    W_np, b_np, y0_np = make_system_np(N)
    mx.set_default_device(mx.cpu)
    W = mx.array(W_np)
    b = mx.array(b_np)
    y0 = mx.array(y0_np)

    def rhs(y, t):
        return (-y + mx.tanh(W @ y + b)) / TAU

    def run():
        sol = ml.integrate.solve(rhs, y0, t_span=T_SPAN, method="dopri5",
                                 atol=ATOL, rtol=RTOL)
        mx.eval(sol.y)
        return sol

    med, std = time_fn(run)
    mx.set_default_device(mx.gpu)
    return med, std


# --------------------------------------------------------------------------- #
# SciPy RK45 = Dormand-Prince 5(4)
# --------------------------------------------------------------------------- #

def bench_scipy(N):
    from scipy.integrate import solve_ivp

    W, b, y0 = make_system_np(N)

    def rhs(t, y):
        return (-y + np.tanh(W @ y + b)) / TAU

    def run():
        return solve_ivp(rhs, T_SPAN, y0, method="RK45",
                         rtol=RTOL, atol=ATOL)

    return time_fn(run)


# --------------------------------------------------------------------------- #
# torchdiffeq dopri5 (MPS GPU)
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
    t_eval = torch.tensor([T_SPAN[0], T_SPAN[1]], device=device)

    class RHS(torch.nn.Module):
        def forward(self, t, y):
            return (-y + torch.tanh(W @ y + b)) / TAU

    func = RHS()

    def run():
        sol = odeint(func, y0, t_eval, method="dopri5", rtol=RTOL, atol=ATOL,
                     options={"dtype": torch.float32})
        torch.mps.synchronize()
        return sol

    return time_fn(run)


# --------------------------------------------------------------------------- #
# Run + save
# --------------------------------------------------------------------------- #

def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_result(framework, N, med, std):
    results = load_results()
    results.setdefault(framework, {})[str(N)] = {"median": med, "std": std}
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def run_framework(name, bench_fn):
    print(f"\n  {name}")
    print(f"  {'-'*50}")
    for N in SIZES:
        med, std = bench_fn(N)
        if med is not None:
            save_result(name, N, med, std)
            print(f"    N={N:>5}: {med:.4f}s +/- {std:.4f}s")
        else:
            print(f"    N={N:>5}: unavailable")


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #

def make_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = load_results()
    if not results:
        print("No results found. Run benchmarks first.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    styles = {
        "mlxlab": {"color": "#FF6B00", "marker": "o", "linewidth": 2.5, "zorder": 10},
        "mlxlab_cpu": {"color": "#FF6B00", "marker": "o", "linewidth": 1.5, "zorder": 8,
                       "linestyle": "--"},
        "scipy": {"color": "#4B8BBE", "marker": "s", "linewidth": 1.5, "zorder": 5},
        "torchdiffeq": {"color": "#EE4C2C", "marker": "^", "linewidth": 1.5, "zorder": 5},
        "julia": {"color": "#9558B2", "marker": "D", "linewidth": 1.5, "zorder": 5},
        "matlab": {"color": "#0076A8", "marker": "v", "linewidth": 1.5, "zorder": 5},
    }

    labels = {
        "mlxlab": "mlxlab dopri5 (MLX GPU)",
        "mlxlab_cpu": "mlxlab dopri5 (MLX CPU)",
        "scipy": "SciPy RK45 (CPU)",
        "torchdiffeq": "torchdiffeq dopri5 (MPS)",
        "julia": "Julia DP5 (CPU)",
        "matlab": "MATLAB ode45 (CPU)",
    }

    for framework, data in results.items():
        sizes = sorted(int(k) for k in data.keys())
        times = [data[str(n)]["median"] for n in sizes]
        stds = [data[str(n)]["std"] for n in sizes]
        style = styles.get(framework, {"color": "gray", "marker": "x", "linewidth": 1})
        label = labels.get(framework, framework)
        ax.errorbar(sizes, times, yerr=stds, label=label,
                    markersize=7, capsize=3, **style)

    ax.set_xlabel("System size N (N×N weight matrix)", fontsize=12)
    ax.set_ylabel("Wall time (seconds)", fontsize=12)
    ax.set_title("Dormand-Prince 5(4) · rate network · rtol=1e-4 · M5 Max", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(n) for n in SIZES])

    out = Path(__file__).parent / "bench_dp5.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    frameworks = {
        "mlxlab": lambda: run_framework("mlxlab", bench_mlxlab),
        "mlxlab_cpu": lambda: run_framework("mlxlab_cpu", bench_mlxlab_cpu),
        "scipy": lambda: run_framework("scipy", bench_scipy),
        "torchdiffeq": lambda: run_framework("torchdiffeq", bench_torchdiffeq),
    }

    if target == "plot":
        make_plot()
    elif target == "all":
        for name, fn in frameworks.items():
            fn()
        make_plot()
    elif target in frameworks:
        frameworks[target]()
    else:
        print(f"Unknown target: {target}. Use: {list(frameworks.keys()) + ['all', 'plot']}")
