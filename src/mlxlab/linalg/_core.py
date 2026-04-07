"""Linear algebra functions built on MLX primitives.

Note: MLX's decompositions (lu, qr, svd, eig, etc.) are CPU-only in MLX 0.31.
All functions here use stream=mx.cpu and eval results immediately to avoid
mixing CPU/GPU computation graphs (which causes bus errors in MLX 0.31).
Results are mx.arrays in unified memory, usable in subsequent GPU operations.
"""

import mlx.core as mx


def _perm_sign_from_vector(perm_list: list[int]) -> float:
    """Sign of a permutation: (-1)^(number of inversions)."""
    sign = 1
    n = len(perm_list)
    for i in range(n):
        for j in range(i + 1, n):
            if perm_list[i] > perm_list[j]:
                sign = -sign
    return float(sign)


def det(a: mx.array) -> mx.array:
    """Determinant of a square matrix via SVD.

    Uses SVD rather than LU because MLX's LU throws an unrecoverable C++
    exception on singular matrices. SVD handles all cases gracefully.
    Sign is recovered from the orthogonal factors.

    Args:
        a: Square matrix (N, N).

    Returns:
        Determinant scalar.
    """
    U, s, Vt = mx.linalg.svd(a, stream=mx.cpu)
    mx.eval(U, s, Vt)

    # |det(A)| = prod(s)
    # sign(det(A)) = det(U) * det(Vt)
    # For orthogonal matrices, det = +/-1
    # det(U) * det(Vt) can be computed as sign of det(U @ Vt)
    # Use the trick: det(orthogonal) = sign of prod of eigenvalues
    # Simpler: det(A) = det(U) * prod(s) * det(Vt)
    # det(U) and det(Vt) are +/-1 each

    import math
    prod_s = 1.0
    for i in range(s.shape[0]):
        prod_s *= float(s[i].item())

    # Compute sign via U @ diag(s) @ Vt vs original
    # Shortcut: compute det(U @ Vt) via the product
    UVt = U @ Vt
    mx.eval(UVt)
    # det of a near-orthogonal matrix: use trace-based approximation
    # or just compute via LU if non-singular, SVD if singular
    # Check smallest singular value — LU crashes (unrecoverable C++ exception)
    # on singular/near-singular matrices, so guard with SVD condition.
    eps = 1.19209e-07  # float32 machine epsilon
    s_min = float(s[-1].item())
    s_max = float(s[0].item()) if s.shape[0] > 0 else 1.0
    if s_min < eps * s_max * a.shape[-1]:
        return mx.array(0.0)

    # Sign: safe to call LU now (matrix is well-conditioned).
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

    Uses SVD for the magnitude and LU for the sign (when non-singular).

    Args:
        a: Square matrix (N, N).

    Returns:
        (sign, logabsdet) where det(a) = sign * exp(logabsdet).
    """
    import math

    s = mx.linalg.svd(a, compute_uv=False, stream=mx.cpu)
    mx.eval(s)

    # Check singularity before LU (LU crashes on singular matrices)
    eps = 1.19209e-07
    s_min = float(s[-1].item())
    s_max = float(s[0].item()) if s.shape[0] > 0 else 1.0
    if s_min < eps * s_max * a.shape[-1]:
        return mx.array(0.0), mx.array(float("-inf"))

    # log|det| = sum(log(s_i))
    logabsdet = 0.0
    for i in range(s.shape[0]):
        logabsdet += math.log(float(s[i].item()))

    # Sign from LU (safe — matrix is non-singular)
    P, L, U = mx.linalg.lu(a, stream=mx.cpu)
    mx.eval(P, U)
    perm = [int(P[i].item()) for i in range(P.shape[0])]
    sign = _perm_sign_from_vector(perm)
    for i in range(U.shape[0]):
        if float(U[i, i].item()) < 0:
            sign *= -1.0
    return mx.array(sign), mx.array(logabsdet)


def lstsq(a: mx.array, b: mx.array) -> mx.array:
    """Least-squares solution to a @ x = b via QR decomposition.

    Args:
        a: Matrix (M, N).
        b: Right-hand side (M,) or (M, K).

    Returns:
        x: Least-squares solution (N,) or (N, K).
    """
    Q, R = mx.linalg.qr(a, stream=mx.cpu)
    mx.eval(Q, R)
    Qtb = Q.T @ b
    mx.eval(Qtb)
    x = mx.linalg.solve_triangular(R, Qtb, upper=True, stream=mx.cpu)
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
        eps = 1.19209e-07  # float32 machine epsilon
        tol = float(max(a.shape[-2], a.shape[-1])) * eps * float(s[0].item())
    count = sum(1 for i in range(s.shape[0]) if float(s[i].item()) > tol)
    return mx.array(count)


def cond(a: mx.array, p=None) -> mx.array:
    """Condition number of a matrix (2-norm) via SVD.

    Args:
        a: Matrix (M, N).

    Returns:
        Condition number (sigma_max / sigma_min).
    """
    s = mx.linalg.svd(a, compute_uv=False, stream=mx.cpu)
    mx.eval(s)
    return mx.array(float(s[0].item()) / float(s[-1].item()))
