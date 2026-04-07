"""Linear algebra utilities built on mlx.core.linalg.

Fills gaps relative to NumPy/SciPy that MLX doesn't ship:
- det, slogdet (from LU)
- lstsq (from QR)
- matrix_rank (from SVD)
- cond (from SVD)

For operations that MLX already provides (eig, svd, cholesky, lu, qr, solve,
inv, pinv, norm, etc.), use mlx.core.linalg directly.
"""

from ._core import det, slogdet, lstsq, matrix_rank, cond

__all__ = ["det", "slogdet", "lstsq", "matrix_rank", "cond"]
