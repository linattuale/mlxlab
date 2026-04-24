"""Mixed-precision helpers for integration RHS functions."""

from collections.abc import Callable

import mlx.core as mx

_DTYPE_ALIASES = {
    "float16": mx.float16,
    "fp16": mx.float16,
    "half": mx.float16,
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
    "float32": mx.float32,
    "fp32": mx.float32,
}


def normalize_dtype(dtype, *, name: str = "dtype"):
    """Normalize string and MLX dtype inputs."""
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
        choices = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(f"Unknown {name} {dtype!r}. Choose from: {choices}.")
    return dtype


def validate_floating_dtype(dtype, *, name: str):
    """Validate that a dtype can be used for floating-point solver work."""
    dtype = normalize_dtype(dtype, name=name)
    if not mx.issubdtype(dtype, mx.floating):
        raise ValueError(f"{name} must be a floating dtype, got {dtype}.")
    return dtype


def wrap_rhs_dtype(f: Callable, *, state_dtype, rhs_dtype) -> Callable:
    """Evaluate f on rhs_dtype state views and cast derivatives back."""
    rhs_dtype = validate_floating_dtype(rhs_dtype, name="rhs_dtype")
    state_dtype = validate_floating_dtype(state_dtype, name="y0.dtype")

    def wrapped(y, t):
        return f(y.astype(rhs_dtype), t).astype(state_dtype)

    return wrapped


def mixed_matmul(
    a: mx.array,
    b: mx.array,
    *,
    dtype=mx.float16,
    out_dtype=mx.float32,
) -> mx.array:
    """Matrix multiply through a low-precision input path.

    Long-lived weights should be pre-cast to ``dtype`` by the caller to avoid
    adding a cast to every RHS evaluation. The result is cast to ``out_dtype`` by
    default so solver state, error estimates, and step-size decisions can remain
    in float32.
    """
    dtype = validate_floating_dtype(dtype, name="dtype")
    result = a.astype(dtype) @ b.astype(dtype)
    if out_dtype is None:
        return result
    return result.astype(validate_floating_dtype(out_dtype, name="out_dtype"))
