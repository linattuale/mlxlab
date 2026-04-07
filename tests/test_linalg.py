"""Tests for mlxlab.linalg.

Note: MLX 0.31 has a bus error when CPU-stream linalg ops (lu, qr, svd) run
under pytest. The functions work correctly in plain Python. We run tests via
subprocess to avoid the crash.
"""

import subprocess
import sys

import pytest


def _run_check(code: str):
    """Run a test snippet in a subprocess, return stdout."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed:\n{result.stderr}")
    return result.stdout.strip()


# ---- det / slogdet ----------------------------------------------------------

def test_det_identity():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml
assert abs(ml.linalg.det(mx.eye(4)).item() - 1.0) < 1e-5
print("OK")
""")
    assert out == "OK"


def test_det_known():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml
assert abs(ml.linalg.det(mx.array([[3.,1.],[2.,4.]])).item() - 10.0) < 1e-2
print("OK")
""")
    assert out == "OK"


def test_det_singular():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml
assert abs(ml.linalg.det(mx.array([[1.,2.],[2.,4.]])).item()) < 1e-3
print("OK")
""")
    assert out == "OK"


def test_slogdet_positive():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml; import math
sign, logabs = ml.linalg.slogdet(mx.array([[2.,0.],[0.,3.]]))
assert abs(sign.item() - 1.0) < 1e-5
assert abs(logabs.item() - math.log(6.0)) < 1e-3
print("OK")
""")
    assert out == "OK"


def test_slogdet_negative():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml
sign, logabs = ml.linalg.slogdet(mx.array([[0.,1.],[1.,0.]]))
assert abs(sign.item() - (-1.0)) < 1e-5
print("OK")
""")
    assert out == "OK"


# ---- lstsq -------------------------------------------------------------------

def test_lstsq_exact():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml
A = mx.array([[1.,2.],[3.,4.]]); x = ml.linalg.lstsq(A, A @ mx.array([1.,2.]))
assert abs(x[0].item() - 1.0) < 1e-3 and abs(x[1].item() - 2.0) < 1e-3
print("OK")
""")
    assert out == "OK"


def test_lstsq_overdetermined():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml; import numpy as np
np.random.seed(42)
A_np = np.random.randn(10, 3).astype(np.float32)
x_true = np.array([1., -2., 0.5], dtype=np.float32)
b_np = A_np @ x_true + 0.01 * np.random.randn(10).astype(np.float32)
x = ml.linalg.lstsq(mx.array(A_np), mx.array(b_np))
for i in range(3):
    assert abs(x[i].item() - x_true[i]) < 0.1, f"x[{i}]={x[i].item()}"
print("OK")
""")
    assert out == "OK"


# ---- matrix_rank / cond ------------------------------------------------------

def test_matrix_rank():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml
assert int(ml.linalg.matrix_rank(mx.eye(2)).item()) == 2
assert int(ml.linalg.matrix_rank(mx.array([[1.,2.],[2.,4.]])).item()) == 1
print("OK")
""")
    assert out == "OK"


def test_cond():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml
assert abs(ml.linalg.cond(mx.eye(3)).item() - 1.0) < 1e-3
assert ml.linalg.cond(mx.array([[1.,0.],[0.,1e-6]])).item() > 1e5
print("OK")
""")
    assert out == "OK"


def test_cond_singular():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml; import math
c = ml.linalg.cond(mx.array([[1.,2.],[2.,4.]])).item()
assert math.isinf(c), f"Expected inf, got {c}"
print("OK")
""")
    assert out == "OK"


def test_det_non_square():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml
try:
    ml.linalg.det(mx.array([[1.,2.,3.],[4.,5.,6.]]))
    print("FAIL")
except ValueError:
    print("OK")
""")
    assert out == "OK"


def test_slogdet_non_square():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml
try:
    ml.linalg.slogdet(mx.array([[1.,2.,3.]]))
    print("FAIL")
except ValueError:
    print("OK")
""")
    assert out == "OK"


def test_lstsq_rank_deficient():
    out = _run_check("""
import mlx.core as mx; import mlxlab as ml; import numpy as np
A = mx.array([[1.,1.],[1.,1.],[1.,1.]])
b = mx.array([2.,2.,2.])
x = ml.linalg.lstsq(A, b)
# Minimum-norm solution: x = [1, 1] (equal split)
assert abs(x[0].item() - 1.0) < 0.1 and abs(x[1].item() - 1.0) < 0.1, f"x={[x[i].item() for i in range(2)]}"
print("OK")
""")
    assert out == "OK"
