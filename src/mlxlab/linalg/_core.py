"""Linear algebra functions built on MLX primitives.

Note: MLX's decompositions (lu, qr, svd, eig, etc.) are CPU-only in MLX 0.31.
All functions here use stream=mx.cpu and eval results immediately to avoid
mixing CPU/GPU computation graphs (which causes bus errors in MLX 0.31).
Results are mx.arrays in unified memory, usable in subsequent GPU operations.
"""

import math

import mlx.core as mx

_EPS32 = 1.19209e-07  # float32 machine epsilon


def _check_square(a: mx.array, name: str):
    if a.ndim < 2 or a.shape[-2] != a.shape[-1]:
        raise ValueError(f"{name} requires a square matrix, got shape {a.shape}.")


def _perm_sign_from_vector(perm_list: list[int]) -> float:
    """Sign of a permutation: (-1)^(number of inversions)."""
    sign = 1
    n = len(perm_list)
    for i in range(n):
        for j in range(i + 1, n):
            if perm_list[i] > perm_list[j]:
                sign = -sign
    return float(sign)


def _is_singular(s: mx.array, n: int) -> bool:
    """Check if a matrix is numerically singular based on SVD singular values."""
    s_min = float(s[-1].item())
    s_max = float(s[0].item()) if s.shape[0] > 0 else 1.0
    return s_min < _EPS32 * s_max * n


def det(a: mx.array) -> mx.array:
    """Determinant of a square matrix.

    Uses SVD for magnitude (handles singular matrices gracefully) and LU for
    sign (only when non-singular, since MLX LU crashes on singular matrices).

    Args:
        a: Square matrix (N, N).

    Returns:
        Determinant scalar.
    """
    _check_square(a, "det")

    U, s, Vt = mx.linalg.svd(a, stream=mx.cpu)
    mx.eval(U, s, Vt)

    prod_s = 1.0
    for i in range(s.shape[0]):
        prod_s *= float(s[i].item())

    if _is_singular(s, a.shape[-1]):
        return mx.array(0.0)

    # Sign from LU (safe — matrix is non-singular)
    P, L, LU_U = mx.linalg.lu(a, stream=mx.cpu)
    mx.eval(P, LU_U)
    perm = [int(P[i].item()) for i in range(P.shape[0])]
    sign = _perm_sign_from_vector(perm)
    lu_sign = 1.0
    for i in range(LU_U.shape[0]):
        if float(LU_U[i, i].item()) < 0:
            lu_sign *= -1.0
    return mx.array(sign * lu_sign * prod_s)


def slogdet(a: mx.array) -> tuple[mx.array, mx.array]:
    """Sign and log-absolute-determinant.

    Args:
        a: Square matrix (N, N).

    Returns:
        (sign, logabsdet) where det(a) = sign * exp(logabsdet).
    """
    _check_square(a, "slogdet")

    s = mx.linalg.svd(a, compute_uv=False, stream=mx.cpu)
    mx.eval(s)

    if _is_singular(s, a.shape[-1]):
        return mx.array(0.0), mx.array(float("-inf"))

    logabsdet = 0.0
    for i in range(s.shape[0]):
        logabsdet += math.log(float(s[i].item()))

    # Sign from LU (safe — non-singular)
    P, L, U = mx.linalg.lu(a, stream=mx.cpu)
    mx.eval(P, U)
    perm = [int(P[i].item()) for i in range(P.shape[0])]
    sign = _perm_sign_from_vector(perm)
    for i in range(U.shape[0]):
        if float(U[i, i].item()) < 0:
            sign *= -1.0
    return mx.array(sign), mx.array(logabsdet)


def lstsq(a: mx.array, b: mx.array) -> mx.array:
    """Least-squares solution to a @ x = b via SVD (pseudoinverse).

    Computes the minimum-norm least-squares solution, matching NumPy/SciPy
    semantics. Handles rank-deficient systems correctly.

    Args:
        a: Matrix (M, N).
        b: Right-hand side (M,) or (M, K).

    Returns:
        x: Minimum-norm least-squares solution (N,) or (N, K).
    """
    # x = pinv(A) @ b gives minimum-norm least-squares
    A_pinv = mx.linalg.pinv(a, stream=mx.cpu)
    mx.eval(A_pinv)
    x = A_pinv @ b
    mx.eval(x)
    return x


def matrix_rank(a: mx.array, tol: float | None = None) -> mx.array:
    """Matrix rank via SVD.

    Args:
        a: Matrix (M, N).
        tol: Threshold. Default: max(M,N) * eps * sigma_max.

    Returns:
        Rank (integer array).
    """
    s = mx.linalg.svd(a, compute_uv=False, stream=mx.cpu)
    mx.eval(s)
    if tol is None:
        tol = float(max(a.shape[-2], a.shape[-1])) * _EPS32 * float(s[0].item())
    count = sum(1 for i in range(s.shape[0]) if float(s[i].item()) > tol)
    return mx.array(count)


def cond(a: mx.array, p=None) -> mx.array:
    """Condition number of a matrix (2-norm) via SVD.

    Returns inf for singular matrices.

    Args:
        a: Matrix (M, N).

    Returns:
        Condition number (sigma_max / sigma_min), or inf if singular.
    """
    s = mx.linalg.svd(a, compute_uv=False, stream=mx.cpu)
    mx.eval(s)
    s_min = float(s[-1].item())
    s_max = float(s[0].item()) if s.shape[0] > 0 else 1.0
    if _is_singular(s, max(a.shape[-2], a.shape[-1])):
        return mx.array(float("inf"))
    return mx.array(s_max / s_min)
